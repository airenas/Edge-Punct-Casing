#!/usr/bin/env python3
import argparse
import logging
import os
from collections import Counter

from tqdm import tqdm

from egs.lt_ai_blkt.local.case import get_case_id, CASE_ID_MAP
from egs.lt_ai_blkt.local.punctuation import get_punctuation_id, PUNCTUATION_ID_MAP
from egs.lt_ai_blkt.local.utils import split_word_punctuation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="""Input text file.
        """,
    )
    return parser.parse_args()


def main():
    args = get_args()

    logging.info(f"info for {args.input}")

    sc, wc, counter_punct, counter_casing, = 0, 0, Counter(), Counter()
    with open(args.input, "r", encoding="utf-8") as text_fp:
        for line in tqdm(text_fp):
            line = line.rstrip("\n")
            if len(line) == 0:
                continue
            sc += 1
            words = line.split()
            for word in words:
                word, punct = split_word_punctuation(word)
                if len(word) == 0:
                    continue
                wc += 1
                pid = get_punctuation_id(punct)
                counter_punct[PUNCTUATION_ID_MAP.get(pid)] += 1

                cid = get_case_id(word)
                counter_casing[CASE_ID_MAP.get(cid)] += 1

    logging.info(f"Total sentences: {sc}")
    logging.info(f"Total words: {wc}")
    logging.info(f"Punctuation distribution: {counter_punct}")
    logging.info(f"Casing distribution: {counter_casing}")
    for k, v in counter_punct.items():
        logging.info(f"Punctuation {k}: {v} ({v/wc:.2%})")
    for k, v in counter_casing.items():
        logging.info(f"Casing {k}: {v} ({v/wc:.2%})")



if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")
