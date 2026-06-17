#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path
from typing import Dict

import sentencepiece as spm
from datasets import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Download HF dataset and export text")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file path, for example /path/to/cleaned.txt.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Vocabulary size for the BPE model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the BPE model.",
    )
    return parser.parse_args()


def generate_tokens(lang_dir: Path):
    """
    Generate the tokens.txt from a bpe model.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(lang_dir / "bpe.model"))
    token2id: Dict[str, int] = {sp.id_to_piece(i): i for i in range(sp.vocab_size())}
    with open(lang_dir / "tokens.txt", "w", encoding="utf-8") as f:
        for sym, i in tqdm(token2id.items()):
            f.write(f"{sym} {i}\n")


def main():
    args = get_args()
    logging.info(f"Training BPE model with vocab size {args.vocab_size} on {args.input}")
    input_sentence_size = 10_000_000
    character_coverage = 1.0
    user_defined_symbols = ["<blk>", "<s>", "</s>"]
    unk_id = len(user_defined_symbols)
    model_type = "unigram"

    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=f"{args.output_dir}/bpe",
        vocab_size=args.vocab_size,
        model_type=model_type,
        input_sentence_size=input_sentence_size,
        character_coverage=character_coverage,
        user_defined_symbols=user_defined_symbols,
        shuffle_input_sentence=True,
        unk_id=unk_id,
        bos_id=-1,
        eos_id=-1,
        train_extremely_large_corpus=True
    )

    generate_tokens(Path(args.output_dir))
    logging.info(f"Trained BPE model")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        format=formatter,
        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING),
    )

    logging.info("Starting")
    main()
    logging.info("Done")
