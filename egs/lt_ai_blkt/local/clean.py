#!/usr/bin/env python3
import argparse
import logging
import os

from tqdm import tqdm

from egs.lt_ai_blkt.local.punctuation import PUNCTUATION
from egs.lt_ai_blkt.local.utils import Word, split_word_punctuation


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


def has_num(word):
    for c in word:
        if c.isdigit():
            return True
    return False


def allowed_symbols(word):
    for c in word:
        cl = c.lower()
        if 'a' <= cl <= 'z':
            continue
        if cl in "ąčęėįšųūž":
            continue
        return False
    return len(word) > 0


def allowed_punctuation(wrd, punct):
    if wrd == "" and punct != "":
        return False
    for p in punct:
        if p not in PUNCTUATION:
                return False
    return True

def skip(w):
    if has_num(w.word):
        return True
    if w.word in ["I", "II", "V", "III", "IV", "VI", "VII", "VIII", "IX", "X"] and w.mi.startswith("M"):
        return True
    if w.mi.startswith("D"):
        return True
    wrd, punct = split_word_punctuation(w.word)
    if not allowed_punctuation(wrd, punct):
        return True
    if not allowed_symbols(wrd):
        return True
    return False


def is_ok(words):
    for i, w in enumerate(words):
        if skip(w):
            return False
    if len(words) > 0:
        last = words[-1].word
        _, p = split_word_punctuation(last)
        if p not in PUNCTUATION:
            return False
    return True


def main():
    args = get_args()

    logging.info(f"fix casing {args.input}")

    logging.info(f"Output file: {args.output}")
    ok, skip = 0, 0
    with open(args.output, "w", encoding="utf-8") as f_out:
        total = os.path.getsize(args.input)
        with open(args.input, "r", encoding="utf-8") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc="Filtering") as pbar:
                for line in f:
                    mc = len(line)
                    line = line.rstrip("\n")
                    strs = line.split()
                    words = [Word(s) for s in strs]
                    if is_ok(words):
                        ok += 1
                        ws = " ".join(w.word for w in words)
                        f_out.write(ws + "\n")
                    else:
                        skip += 1
                    pbar.update(mc)
    logging.info(f"Kept {ok} lines, skipped {skip} lines")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")
