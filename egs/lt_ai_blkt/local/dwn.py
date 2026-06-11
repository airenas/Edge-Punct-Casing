#!/usr/bin/env python3
import argparse
import logging
import math
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import DatasetDict, load_dataset
from tqdm import tqdm


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
        "--num-shards",
        type=int,
        default=0,
        help="Fixed number of output shards. 0 enables dynamic sharding (default: 0).",
    )
    parser.add_argument(
        "--shard-size-mb",
        type=int,
        default=512,
        help="Target shard size in MB for dynamic sharding (default: 512).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        help="Parquet compression codec (default: zstd).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="how many rows to export, 0 means no limit (default: 0)."
    )
    return parser.parse_args()


def transform_text(text: str) -> str:
    # Template hook: place custom text manipulations here.
    return text


class ParquetKeeper:
    def __init__(
        self,
        output_dir: Path,
        base_name: str = "data",
        text_field: str = "text",
        shard_size_mb: int = 512,
        compression: str | None = "zstd",
    ):
        self.output_dir = Path(output_dir)
        self.base_name = base_name
        self.text_field = text_field
        self.shard_size_bytes = max(1, shard_size_mb) * 1024 * 1024
        self.compression = "NONE" if compression in (None, "none") else str(compression)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._writer = None
        self._current_shard_file = None
        self._rows_in_current_shard = 0
        self._shard_idx = 0
        self.rows_written = 0

    @property
    def shard_count(self) -> int:
        return self._shard_idx

    def _shard_file(self, shard_idx: int) -> Path:
        return self.output_dir / f"{self.base_name}-{shard_idx:05d}.parquet"

    def _open_current_shard(self):
        """Open a ParquetWriter for the current shard index."""
        shard_file = self._shard_file(self._shard_idx)
        schema = pa.schema([(self.text_field, pa.string())])
        self._writer = pq.ParquetWriter(
            str(shard_file),
            schema,
            compression=self.compression,
        )
        self._current_shard_file = shard_file
        logging.info("Opened shard %d for writing: %s", self._shard_idx, shard_file)

    def _get_current_file_size(self) -> int:
        """Get the current shard file size in bytes."""
        if self._current_shard_file and self._current_shard_file.exists():
            return self._current_shard_file.stat().st_size
        return 0

    def _should_close_shard(self) -> bool:
        """Check if current shard should be closed and a new one opened."""
        if self._get_current_file_size() >= self.shard_size_bytes:
            return True
        return False

    def _close_current_shard(self):
        """Close the current ParquetWriter and move to next shard."""
        if self._writer:
            self._writer.close()
            logging.info(
                "Closed shard %d with %d rows: %s",
                self._shard_idx,
                self._rows_in_current_shard,
                self._current_shard_file,
            )
            self._writer = None
            self._shard_idx += 1
            self._rows_in_current_shard = 0
            self._current_shard_file = None

    def feed_text(self, text: str) -> bool:
        """Feed a single text string to the writer (writes to current or new shard)."""
        if not isinstance(text, str):
            return False

        # Open first shard if needed
        if self._writer is None:
            self._open_current_shard()

        # Write single row as a table
        table = pa.Table.from_arrays(
            [pa.array([text], type=pa.string())],
            names=[self.text_field],
        )
        self._writer.write_table(table)
        self._rows_in_current_shard += 1
        self.rows_written += 1

        # Close shard and open next if size/row limit reached
        if self._should_close_shard():
            self._close_current_shard()

        return True

    def restore_shard_index(self, shard_idx: int):
        """Restore shard counter to continue from next shard (e.g., on resumption)."""
        self._shard_idx = shard_idx + 1
        logging.info("Restored shard counter to %d (will write to shard %d next)", shard_idx, self._shard_idx)

    def close(self):
        """Close the current shard writer."""
        self._close_current_shard()


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

    compression = args.compression

    if args.num_shards < 0:
        raise ValueError("--num-shards must be >= 0")

    output_path = Path(args.output)
    if output_path.suffix == ".parquet":
        output_dir = output_path.parent
        base_name = output_path.stem
    else:
        output_dir = output_path
        base_name = "data"

    max_rows_per_shard = None
    if args.num_shards > 0:
        total_for_split = args.limit if args.limit > 0 else ds.num_rows
        max_rows_per_shard = max(1, math.ceil(total_for_split / args.num_shards))

    keeper = ParquetKeeper(
        output_dir=output_dir,
        base_name=base_name,
        text_field=args.text_field,
        shard_size_mb=args.shard_size_mb,
        compression=compression,
    )

    rows_seen = 0
    rows_written = 0
    rows_skipped = 0
    for row in tqdm(ds):
        value = row.get(args.text_field)
        if not isinstance(value, str):
            rows_skipped += 1
            continue

        value = transform_text(value)
        if value is None:
            rows_skipped += 1
            continue

        if keeper.feed_text(value):
            rows_seen += 1
            rows_written += 1

        if args.limit > 0 and rows_seen >= args.limit:
            logging.info("Reached limit of %d rows, stopping", args.limit)
            break

    keeper.close()

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
