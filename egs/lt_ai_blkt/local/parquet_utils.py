import logging
from pathlib import Path
from typing import List

import pyarrow.parquet as pq
import pyarrow as pa

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
        self._current_tmp_file = None
        self._rows_in_current_shard = 0
        self._shard_idx = 0
        self.rows_written = 0

        self._buffer = []
        self._batch_size = 5000

    @property
    def shard_count(self) -> int:
        return self._shard_idx

    def _shard_file(self, shard_idx: int) -> Path:
        return self.output_dir / f"{self.base_name}-{shard_idx:05d}.parquet"

    def _tmp_shard_file(self, shard_idx: int) -> Path:
        return self.output_dir / f"{self.base_name}-{shard_idx:05d}.parquet.inprogress"

    def _open_current_shard(self):
        """Open a ParquetWriter for the current shard index."""
        shard_file = self._shard_file(self._shard_idx)
        tmp_file = self._tmp_shard_file(self._shard_idx)
        schema = pa.schema([(self.text_field, pa.string())])
        self._writer = pq.ParquetWriter(
            str(tmp_file),
            schema,
            compression=self.compression,
        )
        self._current_shard_file = shard_file
        self._current_tmp_file = tmp_file
        logging.info("Opened shard %d for writing: %s", self._shard_idx, shard_file)

    def _get_current_file_size(self) -> int:
        """Get the current shard file size in bytes."""
        if self._current_tmp_file and self._current_tmp_file.exists():
            return self._current_tmp_file.stat().st_size
        return 0

    def _should_close_shard(self) -> bool:
        """Check if current shard should be closed and a new one opened."""
        if self._get_current_file_size() >= self.shard_size_bytes:
            return True
        return False

    def _close_current_shard(self):
        """Close the current ParquetWriter and move to next shard."""
        if self._writer:
            self._flush()
            self._writer.close()
            if self._current_tmp_file and self._current_shard_file:
                self._current_tmp_file.replace(self._current_shard_file)
            logging.info(
                "Closed shard %d with %d rows: %s",
                self._shard_idx,
                self._rows_in_current_shard,
                self._current_shard_file,
            )
            self._writer = None
            self._shard_idx += 1
            self._rows_in_current_shard = 0
            self._current_tmp_file = None
            self._current_shard_file = None

    def feed_text(self, text: str) -> bool:
        if not isinstance(text, str):
            return False

        if self._writer is None:
            self._open_current_shard()

        self._buffer.append(text)

        if len(self._buffer) >= self._batch_size:
            self._flush()
            if self._should_close_shard():
                self._close_current_shard()

        return True

    def _flush(self):
        if not self._buffer:
            return

        table = pa.Table.from_arrays(
            [pa.array(self._buffer, type=pa.string())],
            names=[self.text_field],
        )

        self._writer.write_table(table)

        n = len(self._buffer)
        self._rows_in_current_shard += n
        self.rows_written += n

        self._buffer.clear()


    def restore_shard_index(self, shard_idx: int):
        """Restore shard counter to continue from next shard (e.g., on resumption)."""
        self._shard_idx = shard_idx + 1
        logging.info("Restored shard counter to %d (will write to shard %d next)", shard_idx, self._shard_idx)

    def close(self):
        """Close the current shard writer."""
        self._close_current_shard()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _resolve_parquet_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if files:
            return files
    raise ValueError(f"No parquet files found in '{input_path}'")


def iter_text_rows(input_path: str, text_field: str = "text"):
    for parquet_file in _resolve_parquet_files(input_path):
        pf = pq.ParquetFile(parquet_file)
        if text_field not in set(pf.schema_arrow.names):
            raise ValueError(
                f"Column '{text_field}' not found in {parquet_file}. Available: {', '.join(pf.schema_arrow.names)}"
            )
        for batch in pf.iter_batches(columns=[text_field]):
            for value in batch.column(0).to_pylist():
                if isinstance(value, str):
                    yield value


def count_rows(input_path: str) -> int:
    total = 0
    for parquet_file in _resolve_parquet_files(input_path):
        total += pq.ParquetFile(parquet_file).metadata.num_rows
    return total
