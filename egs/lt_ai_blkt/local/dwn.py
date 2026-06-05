#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path

from datasets import DatasetDict, load_dataset
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Download HF dataset and export text")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Hugging Face dataset id, for example VSSA-SDSA/LT_AI_BLKT.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field/column that contains document text (default: text).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path, for example /path/to/lt.txt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="how many rows to export, 0 means no limit (default: 0)."
    )
    return parser.parse_args()


def main():
    args = get_args()
    out_file = Path(args.output)

    logging.info("Loading dataset %s", args.dataset)
    ds_loaded = load_dataset(args.dataset)
    if isinstance(ds_loaded, DatasetDict):
        split_name = next(iter(ds_loaded.keys()))
        logging.info("Using split '%s'", split_name)
        ds = ds_loaded[split_name]
    else:
        ds = ds_loaded

    if args.text_field not in ds.column_names:
        raise ValueError(
            f"Column '{args.text_field}' not found. Available: {', '.join(ds.column_names)}"
        )

    rows_written = 0
    with out_file.open("w", encoding="utf-8") as f_out:
        for row in tqdm(ds, desc="Exporting", unit="rows"):
            value = row.get(args.text_field)
            if not isinstance(value, str):
                continue

            # Keep text exactly as provided by the dataset.
            f_out.write(value)
            if not value.endswith("\n"):
                f_out.write("\n")
            rows_written += 1
            if 0 < args.limit <= rows_written:
                logging.info("Reached limit of %d rows, stopping", args.limit)
                break

    logging.info("Wrote %d rows to %s", rows_written, out_file)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        format=formatter,
        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING),
    )

    logging.info("Starting")
    main()
    logging.info("Done")
