#!/usr/bin/env python3
import argparse
import logging
import os

from egs.lt_ai_blkt.local.data_module import TextDatasetNew
import sentencepiece as spm


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
        help="""Output features file
            """,
    )
    parser.add_argument("--bpe_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The bpe model path")
    parser.add_argument("--max_seq_length",
                        default=200,
                        type=int,
                        help="The sequence length of one sample after SentencePiece tokenization")
    return parser.parse_args()


def main():
    args = get_args()

    logging.info(f"fix casing {args.input}")

    logging.info(f"Output file: {args.output}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    tds = TextDatasetNew(tokenizer=sp)
    tds.convert_examples_to_features(args.max_seq_length, args.input, args.output)
    logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")
