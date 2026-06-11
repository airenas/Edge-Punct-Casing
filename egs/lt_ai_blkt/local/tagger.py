#!/usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List

import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from egs.lt_ai_blkt.local.dwn import ParquetKeeper
from egs.lt_ai_blkt.local.utils import has_upper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="""Input parquet file or directory with parquet shards.
        """,
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="""Input parquet column that contains text.
        """,
    )
    parser.add_argument(
        "--tagger-url",
        type=str,
        required=True,
        help="""Tagger URL.
        """,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="""Output parquet path (file name prefix or .parquet file name).
            """,
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        help="""Parquet compression codec (default: zstd).
        """,
    )
    parser.add_argument(
        "--shard-size-mb",
        type=int,
        default=512,
        help="""Target shard size in MB (default: 512).
        """,
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=".tagger_last_conversion.json",
        help="""Path to conversion state json file.
        """,
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="""Restart conversion from scratch (clear prior output shards and state).
        """,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="""How many input rows between state checkpoints (default: 100).
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


def write_out(keeper: ParquetKeeper, sentences) -> int:
    for s in sentences:
        keeper.feed_text(s)
    return len(sentences)


def _resolve_parquet_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if files:
            return files
    raise ValueError(f"No parquet files found in '{input_path}'")


def _iter_text_rows(input_path: str, text_field: str):
    for parquet_file in _resolve_parquet_files(input_path):
        pf = pq.ParquetFile(parquet_file)
        schema_names = set(pf.schema_arrow.names)
        if text_field not in schema_names:
            raise ValueError(
                f"Column '{text_field}' not found in {parquet_file}. Available: {', '.join(pf.schema_arrow.names)}"
            )
        for batch in pf.iter_batches(columns=[text_field]):
            for value in batch.column(0).to_pylist():
                if isinstance(value, str):
                    yield value


def _count_rows(input_path: str) -> int:
    total = 0
    for parquet_file in _resolve_parquet_files(input_path):
        total += pq.ParquetFile(parquet_file).metadata.num_rows
    return total


def _make_keeper(output_path: str, text_field: str, compression: str, shard_size_mb: int) -> ParquetKeeper:
    output = Path(output_path)
    if output.suffix == ".parquet":
        output_dir = output.parent
        base_name = output.stem
    else:
        output_dir = output
        base_name = "data"

    return ParquetKeeper(
        output_dir=output_dir,
        base_name=base_name,
        text_field=text_field,
        shard_size_mb=shard_size_mb,
        compression=compression,
    )


def _resolve_output_parts(output_path: str):
    output = Path(output_path)
    if output.suffix == ".parquet":
        return output.parent, output.stem
    return output, "data"


def _remove_existing_output_shards(output_path: str):
    output_dir, base_name = _resolve_output_parts(output_path)
    if not output_dir.exists():
        return
    for shard in output_dir.glob(f"{base_name}-*.parquet"):
        shard.unlink()


def _load_state(state_file: str):
    path = Path(state_file)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state_file: str, state: dict):
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)


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
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be > 0")

    previous_state = _load_state(args.state_file)
    if previous_state:
        logging.info("Last conversion state found: %s", args.state_file)
        logging.info(
            "Last conversion: status=%s input=%s output=%s finished_at=%s",
            previous_state.get("status", "unknown"),
            previous_state.get("input", ""),
            previous_state.get("output", ""),
            previous_state.get("finished_at", ""),
        )

    if args.restart:
        logging.info("Restart requested, removing previous output shards and resetting state")
        _remove_existing_output_shards(args.output)
        state_path = Path(args.state_file)
        if state_path.exists():
            state_path.unlink()

    logging.info(f"Tagging parquet text from {args.input}")

    data = Data()
    wrote = 0

    logging.info(f"Output parquet: {args.output}")
    keeper = _make_keeper(
        output_path=args.output,
        text_field=args.text_field,
        compression=args.compression,
        shard_size_mb=args.shard_size_mb,
    )

    sc, wc = 0, 0
    total_rows = _count_rows(args.input)
    started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    state = {
        "status": "running",
        "started_at": started_at,
        "finished_at": "",
        "input": args.input,
        "output": args.output,
        "text_field": args.text_field,
        "processed_rows": 0,
        "read_rows": 0,
        "sentences_written": 0,
        "words_written": 0,
        "output_shard_count": 0,
        "total_rows": total_rows,
    }
    _save_state(args.state_file, state)

    resume_from_row = 0
    resume_shard_idx = 0
    if previous_state and previous_state.get("status") == "running" and not args.restart:
        resume_from_row = previous_state.get("processed_rows", 0)
        resume_shard_idx = previous_state.get("output_shard_count", 0)
        logging.info("Resuming from input row %d, output shard %d", resume_from_row, resume_shard_idx)
        if resume_from_row > 0:
            keeper.restore_shard_index(resume_shard_idx)

    last_idx = 0
    try:
        with tqdm(total=total_rows, unit="rows", desc="Tagging rows") as pbar:
            for idx, line in enumerate(_iter_text_rows(args.input, args.text_field), start=1):
                if idx <= resume_from_row:
                    pbar.update(1)
                    continue

                last_idx = idx
                data.append(line)

                ll = len(data.buffer)
                while ll > MAX_CHARS:
                    sentences = split(args.tagger_url, data, False)
                    sc += len(sentences)
                    wc += sum(len(s.split()) for s in sentences)
                    wrote += write_out(keeper, sentences)
                    pbar.set_postfix(sentences=sc, words=wc)
                    if ll == len(data.buffer):
                        logging.warning(
                            f"Buffer length did not decrease after splitting, breaking to avoid infinite loop. Buffer content: '{data.buffer[:100]}'"
                        )
                        data.buffer = ""
                    ll = len(data.buffer)

                pbar.update(1)
                pbar.set_postfix(sentences=sc, words=wc)
                if idx % args.checkpoint_every == 0:
                    state["processed_rows"] = idx
                    state["output_shard_count"] = keeper.shard_count
                    state["read_rows"] = data.read_lines
                    state["sentences_written"] = wrote
                    state["words_written"] = wc
                    _save_state(args.state_file, state)

            ll = len(data.buffer)
            while ll > MAX_CHARS:
                sentences = split(args.tagger_url, data, True)
                wrote += write_out(keeper, sentences)
                if ll == len(data.buffer):
                    logging.warning(
                        f"Buffer length did not decrease after splitting, breaking to avoid infinite loop. Buffer content: '{data.buffer[:100]}'"
                    )
                    data.buffer = ""
                ll = len(data.buffer)

            if data.buffer:
                sentences = split(args.tagger_url, data, True)
                sc += len(sentences)
                wc += sum(len(s.split()) for s in sentences)
                wrote += write_out(keeper, sentences)
                pbar.set_postfix(sentences=sc, words=wc)

        keeper.close()
        state["status"] = "completed"
        state["finished_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        state["processed_rows"] = last_idx
        state["output_shard_count"] = keeper.shard_count
        state["read_rows"] = data.read_lines
        state["sentences_written"] = wrote
        state["words_written"] = wc
        _save_state(args.state_file, state)
        logging.info(f"read {data.read_lines}, wrote {wrote} sentences")
    except Exception:
        keeper.flush()
        state["status"] = "failed"
        state["finished_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        state["processed_rows"] = last_idx
        state["output_shard_count"] = keeper.shard_count
        state["read_rows"] = data.read_lines
        state["sentences_written"] = wrote
        state["words_written"] = wc
        _save_state(args.state_file, state)
        raise


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")
