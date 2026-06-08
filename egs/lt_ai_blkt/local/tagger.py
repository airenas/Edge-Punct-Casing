#!/usr/bin/env python3
import argparse
import logging
import os
from typing import List

import requests
from tqdm import tqdm

from egs.lt_ai_blkt.local.utils import has_upper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="""Input text file.
        """,
    )
    parser.add_argument(
        "--tagger-url",
        type=str,
        help="""Tagger URL.
        """,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="""Output file
            """,
    )
    return parser.parse_args()


MAX_CHARS = 10000


class Data:
    def __init__(self):
        self.buffer: str = ""
        self.read_lines = 0

    def append(self, line: str):
        self.buffer += line + "\n"
        self.read_lines += 1

    def take_sentences(self, resp, is_last) -> List[str]:
        res = []
        sent = ""
        take_chars = 0
        up_to = 0
        rl = len(resp)
        for i, w in enumerate(resp):
            tp = w.get("type")
            if tp == "SENTENCE_END":
                if sent:
                    up_to = take_chars
                    res.append(sent.strip())
                    sent = ""
                continue
            s = w.get("string", "")
            take_chars += len(s)
            if tp == "WORD":
                mi = w.get("mi", "")
                if has_upper(s):
                    s = s + f"(={w.get("mi", "")})"
                elif mi == "Dl" or mi == "De":
                    logging.info(f"Got word: '{w}'")
                    s = s + f"(={mi})"
            elif tp == "SEPARATOR":
                s = s.replace("\n", " ")
            elif tp == "SPACE":
                pass
            elif tp == "NUMBER":
                pass
            else:
                raise RuntimeError(f"Unknown type: {tp}")
            sent += s
            if not is_last and i > rl * 0.95:
                break
        self.buffer = self.buffer[up_to:]
        return res


def write_out(f_out, sentences) -> int:
    for s in sentences:
        f_out.write(s + "\n")
    return len(sentences)


def call(url: str, txt):
    logging.debug(f"Calling tagger with text of length {len(txt)}")
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(url, data=txt.encode("utf-8"), headers=headers, timeout=30)
    except requests.RequestException as err:
        raise RuntimeError(f"failed to send request: {err}") from err

    logging.debug("resp'%s'", resp.status_code)
    if resp.ok:
        try:
            resp_json = resp.json()
        except ValueError as err:
            raise RuntimeError(f"failed to deserialize response: {err}") from err
        return resp_json

    body_str = resp.content.decode("utf-8", errors="replace")
    raise RuntimeError(f"Failed to make request: {resp.status_code} {body_str}")


def split(url: str, data, is_last):
    txt = data.buffer[:MAX_CHARS]
    res = call(url, txt)
    # logging.debug(res)
    return data.take_sentences(res, is_last)


def main():
    args = get_args()

    logging.info(f"splitting to sentences {args.input}")

    data = Data()
    wrote = 0

    logging.info(f"Output file: {args.output}")
    cc, sc, wc = 0, 0, 0
    with open(args.output, "w", encoding="utf-8") as f_out:
        total = os.path.getsize(args.input)
        with open(args.input, "r", encoding="utf-8") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc="Splitting into sentences") as pbar:
                for line in f:
                    line = line.rstrip("\n")
                    cc += len(line.encode("utf-8"))
                    data.append(line)

                    ll = len(data.buffer)
                    while ll > MAX_CHARS:
                        sentences = split(args.tagger_url, data, False)
                        sc += len(sentences)
                        wc += sum(len(s.split()) for s in sentences)
                        wrote += write_out(f_out, sentences)
                        pos = cc - len(data.buffer)
                        pbar.set_postfix(sentences=sc, words=wc)
                        pbar.update(pos - pbar.n)
                        if ll == len(data.buffer):
                            logging.warning(
                                f"Buffer length did not decrease after splitting, breaking to avoid infinite loop. Buffer content: '{data.buffer[:100]}'")
                            break
                        ll = len(data.buffer)

                    pos = cc - len(data.buffer)
                    pbar.set_postfix(sentences=sc, words=wc)
                    pbar.update(pos - pbar.n)

            # send remaining data
            ll = len(data.buffer)
            while ll > MAX_CHARS:
                sentences = split(args.tagger_url, data, True)
                wrote += write_out(f_out, sentences)
                if ll == len(data.buffer):
                    logging.warning(
                        f"Buffer length did not decrease after splitting, breaking to avoid infinite loop. Buffer content: '{data.buffer[:100]}'")
                    break
                ll = len(data.buffer)
        logging.info(f"read {data.read_lines}, wrote {wrote} sentences")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")
