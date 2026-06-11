#!/usr/bin/env python3
import argparse
import logging
import os

from tqdm import tqdm

from egs.lt_ai_blkt.local.utils import split_word_punctuation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="""Input text file.
        """,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="""Output file
            """,
    )
    return parser.parse_args()


def main():
    args = get_args()

    logging.info(f"to lowercase {args.input}")

    logging.info(f"Output file: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f_out:
        total = os.path.getsize(args.input)
        with open(args.input, "r", encoding="utf-8") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc="To lowercase") as pbar:
                for line in f:
                    mc = len(line)
                    line = line.rstrip("\n")
                    strs = line.split()
                    res = []
                    for s in strs:
                        w, _ = split_word_punctuation(s)
                        if len(w):
                            res.append(w.lower())
                    ws = " ".join(w for w in res)
                    f_out.write(ws + "\n")
                    pbar.update(mc)
        logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")
