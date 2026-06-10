#!/usr/bin/env python3
import argparse
import logging
import os

from tqdm import tqdm

from egs.lt_ai_blkt.local.utils import several_upper, Word


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

    logging.info(f"Output file: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f_out:
        total = os.path.getsize(args.input)
        with open(args.input, "r", encoding="utf-8") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc="Fix casing") as pbar:
                for line in f:
                    mc = len(line)
                    line = line.rstrip("\n")
                    strs = line.split()
                    words = [Word(s) for s in strs]
                    for i, w in enumerate(words):
                        fix_case(w, i)
                    ws = " ".join(w.to_str() for w in words)
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
