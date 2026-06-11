#!/usr/bin/env python3
import argparse
import logging
import os

from tqdm import tqdm

from egs.lt_ai_blkt.local.dwn import ParquetKeeper
from egs.lt_ai_blkt.local.parquet_utils import count_rows, iter_text_rows
from egs.lt_ai_blkt.local.utils import several_upper, Word


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
        help="""Input parquet column that contains text (also used for output).
        """,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="""Output parquet path (file name prefix or .parquet file name).
            """,
    )
    return parser.parse_args()


def is_number(w):
    if w.mi.startswith("M"):
        return True
    return False


def is_abbr(w):
    if w.mi.startswith("X"):
        return True
    if w.mi.startswith("Ya"):
        return True
    return False


def is_proper(w):
    if w.mi.startswith("Np"):
        return True
    return False


def fix_case(w, i):
    if several_upper(w.word):
        old = w.word
        if is_number(w):
            pass
        elif is_abbr(w):
            pass
        elif is_proper(w):
            w.word = w.word.title()
        elif i == 0:
            w.word = w.word.title()
        else:
            w.word = w.word.lower()
        # logging.info(f"Fix case {old}->{w.word}")


def main():
    args = get_args()

    logging.info(f"fix casing {args.input}")

    logging.info(f"Output parquet: {args.output}")

    total = count_rows(args.input)
    wrote = 0
    with tqdm(total=total, unit="rows", desc="Fix casing") as pbar:
        with ParquetKeeper(output_dir=args.output, text_field=args.text_field) as keeper:
            for line in iter_text_rows(args.input, args.text_field):
                strs = line.split()
                words = [Word(s) for s in strs]
                for i, w in enumerate(words):
                    fix_case(w, i)
                ws = " ".join(w.to_str() for w in words)
                if keeper.feed_text(ws):
                    wrote += 1
                pbar.update(1)

    logging.info("Done, wrote %d rows", wrote)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")
