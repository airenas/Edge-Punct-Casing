#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path

from datasets import DatasetDict, load_dataset
from tqdm import tqdm

from egs.lt_ai_blkt.local.parquet_utils import ParquetKeeper


def get_args():
    parser = argparse.ArgumentParser(description="Download HF dataset and export Parquet")
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
        help="Output Parquet file path, for example /path/to/lt.parquet.",
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

    output_path = Path(args.output)

    rows_seen = 0
    rows_written = 0
    rows_skipped = 0
    with ParquetKeeper(output_dir=output_path, text_field=args.text_field) as keeper:
        for row in tqdm(ds):
            value = row.get(args.text_field)
            if not isinstance(value, str):
                rows_skipped += 1
                continue

            if value is None:
                rows_skipped += 1
                continue

            if keeper.feed_text(value):
                rows_seen += 1
                rows_written += 1

            if args.limit > 0 and rows_seen >= args.limit:
                logging.info("Reached limit of %d rows, stopping", args.limit)
                break

    logging.info(
        "Done. Seen=%d, written=%d, skipped=%d, shards=%d",
        rows_seen,
        rows_written,
        rows_skipped,
        keeper.shard_count,
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        format=formatter,
        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING),
    )

    logging.info("Starting")
    main()
    logging.info("Done")
