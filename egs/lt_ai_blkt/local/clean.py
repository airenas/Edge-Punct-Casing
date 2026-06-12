#!/usr/bin/env python3
import argparse
import logging
import os
from typing import List

from tqdm import tqdm

from egs.lt_ai_blkt.local.parquet_utils import count_rows, ParquetKeeper, iter_text_rows
from egs.lt_ai_blkt.local.punctuation import PUNCTUATION, DASH
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
    if punct not in PUNCTUATION:
        return False
    return True


def skip(w: Word):
    if has_num(w.word):
        return True
    if w.word in ["I", "II", "V", "III", "IV", "VI", "VII", "VIII", "IX", "X"] and w.mi.startswith("M"):
        return True
    if w.mi.startswith("D"):
        return True
    wrd, punct = w.word, w.punct
    if not allowed_punctuation(wrd, punct):
        return True
    if not allowed_symbols(wrd):
        return True
    return False


def is_ok(words: List[Word]):
    for i, w in enumerate(words):
        if skip(w):
            return False
    if len(words) > 0:
        first = words[0].word
        if len(first) > 0 and first[0].islower():
            return False
        last = words[-1]
        if last.punct == "":
            return False
    return True


def fix_punctuation(punct):
    for c in punct:
        if c in _clean:
            continue
        if c in PUNCTUATION:
            return c
        if c == " ":
            continue
        if c in "-—‐‑‒":
            return DASH
    return punct


def fix_symbols(words: List[Word]):
    '''Drop quotes and parentheses, remove several punctuations symbols'''
    res = []
    last = None
    for w in words:
        punct = fix_punctuation(w.punct)
        w.punct = punct
        if w.word == "" and punct != "": # add to previous word or drop
            if last is not None:
                if last.word != "" and last.punct == "":
                    last.punct = punct
                continue
        last = w
        res.append(w)
    return res

_clean = "'ˈʼ′´ꞌꞋ`ʽ‘\"'`”‟¨″•"

def drop_symbols(line):
    res = []
    last = ""
    for c in line:
        if c in _clean:
            c = " "
        if c == " " and last == " ":
            continue
        last = c
        res.append(c)
    return "".join(res).strip()


def main():
    args = get_args()

    logging.info(f"fix casing {args.input}")

    logging.info(f"Output file: {args.output}")
    ok, skip = 0, 0

    total = count_rows(args.input)
    with tqdm(total=total, unit="rows", desc="Filtering") as pbar:
        with ParquetKeeper(output_dir=args.output) as keeper:
            for line in iter_text_rows(args.input, keeper.text_field):
                line = line.rstrip("\n")
                line = drop_symbols(line)
                strs = line.split()
                words = [Word(s) for s in strs]
                words = fix_symbols(words)
                if is_ok(words):
                    ok += 1
                    ws = " ".join(w.word + w.punct for w in words)
                    keeper.feed_text(ws)
                else:
                    skip += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok, skip=skip)
    logging.info(f"Kept {ok} lines, skipped {skip} lines")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")
