from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from egs.lt_ai_blkt.local.parquet_utils import ParquetKeeper, count_rows


class TestParquetKeeper(TestCase):
    def test_rotates_shards_after_size_threshold(self):
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            keeper = ParquetKeeper(output_dir=output_dir, shard_size_mb=1)
            keeper._batch_size = 1
            keeper.shard_size_bytes = 1

            self.assertTrue(keeper.feed_text("first row"))
            self.assertTrue(keeper.feed_text("second row"))
            keeper.close()

            shard_files = sorted(output_dir.glob("*.parquet"))
            self.assertEqual(len(shard_files), 2)
            self.assertEqual(keeper.shard_count, 2)
            self.assertEqual(count_rows(str(output_dir)), 2)
