#!/usr/bin/env python3
import argparse
import json
import logging
import os

import pyarrow.parquet as pq


def get_args():
    parser = argparse.ArgumentParser(description="Inspect feature parquet file/schema safely")
    parser.add_argument("--input", type=str, required=True, help="Path to .parquet file")
    parser.add_argument("--rows", type=int, default=2, help="How many sample rows to print")
    return parser.parse_args()


def _short_row(row):
    return {
        "label_len": row["label_len"],
        "token_ids_len": len(row["token_ids"]),
        "case_labels_len": len(row["case_labels"]),
        "punct_labels_len": len(row["punct_labels"]),
        "valid_ids_len": len(row["valid_ids"]),
        "token_masks_len": len(row["token_masks"]),
        "label_masks_len": len(row["label_masks"]),
        "token_ids_head": row["token_ids"][:12],
        "case_labels_head": row["case_labels"][:12],
        "punct_labels_head": row["punct_labels"][:12],
    }


def main():
    args = get_args()

    pf = pq.ParquetFile(args.input)
    print("file:", args.input)
    print("num_row_groups:", pf.num_row_groups)
    print("num_rows:", pf.metadata.num_rows)
    print("schema:")
    print(pf.schema_arrow)

    rows_to_show = max(0, args.rows)
    if rows_to_show == 0:
        return

    shown = 0
    for batch in pf.iter_batches(
        columns=[
            "token_ids",
            "case_labels",
            "punct_labels",
            "valid_ids",
            "token_masks",
            "label_masks",
            "label_len",
        ],
        batch_size=64,
    ):
        table = batch.to_pydict()
        n = len(table["label_len"])
        for i in range(n):
            row = {k: table[k][i] for k in table}
            print(f"row[{shown}]:")
            print(json.dumps(_short_row(row), ensure_ascii=True))
            shown += 1
            if shown >= rows_to_show:
                return


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        format=formatter,
        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING),
    )
    main()
