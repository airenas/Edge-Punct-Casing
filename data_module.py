import argparse
import logging
import random

import numpy as np
import pyarrow.parquet as pq
import sentencepiece
from torch.utils.data import (Dataset, DataLoader, IterableDataset, get_worker_info)
from tqdm import tqdm

from egs.lt_ai_blkt.local.data_module import InputFeatures


class TextDataset(Dataset):

    def __init__(self):
        self.features = []

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            logging.info(f"idx:{idx}, idx.start:{idx.start}, idx.stop:{idx.stop}, idx.step:{idx.step}")
            logging.info(self.features[idx])
            for i in range(*idx.indices(len(self))):
                logging.info(i)
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
        else:
            token_ids = self.features[idx].token_ids
            label_ids = self.features[idx].label_ids
            valid_ids = self.features[idx].valid_ids
            label_len = self.features[idx].label_len
            label_masks = self.features[idx].label_masks
            # print(f"getitem, token_ids:{token_ids}")
            # print(f"getitem {idx}")
            return np.array(token_ids), np.array(label_ids), np.array(valid_ids), label_len, np.array(label_masks)

    def load_features(self, filename, max_seq_length):
        self.features = []
        with open(filename, "r") as fp:
            lines = fp.readlines()
            indx = 0
            tokens = []
            labels = [[], []]
            valid = []
            token_masks = []
            label_masks = []
            label_len = 0
            for i, line in enumerate(tqdm(lines)):
                numbers = line.split()
                n_list = [int(n) for n in numbers]
                if indx == 0:
                    tokens = n_list
                elif indx == 1:
                    labels[0] = n_list
                elif indx == 2:
                    labels[1] = n_list
                elif indx == 3:
                    valid = n_list
                elif indx == 4:
                    token_masks = n_list
                elif indx == 5:
                    label_masks = n_list
                elif indx == 6:
                    assert len(n_list) == 1
                    label_len = n_list[0]

                # print(f"len(tokens):{len(tokens)}, len(token_masks):{len(token_masks)}")
                indx += 1
                if indx == 7:
                    assert len(tokens) == max_seq_length
                    assert len(token_masks) == max_seq_length
                    assert len(valid) == max_seq_length
                    assert len(labels[0]) == max_seq_length
                    assert len(labels[1]) == max_seq_length
                    assert len(label_masks) == max_seq_length
                    assert (label_len > 0 & label_len <= 200)

                    self.features.append(InputFeatures(token_ids=tokens,
                                                       label_ids=labels,
                                                       valid_ids=valid,
                                                       token_masks=token_masks,
                                                       label_masks=label_masks,
                                                       label_len=label_len))
                    indx = 0

                    tokens = []
                    labels = [[], []]
                    valid = []
                    token_masks = []
                    label_masks = []
                    label_len = 0


class StreamingParquetDataset(IterableDataset):
    def __init__(
            self,
            filename: str,
            max_seq_length: int,
            world_size: int = 1,
            rank: int = 0,
            shuffle_buffer_size: int = 10000,
            seed: int = 42,
    ):
        super().__init__()
        self.filename = str(filename)
        self.max_seq_length = max_seq_length
        self.world_size = world_size
        self.rank = rank
        self.shuffle_buffer_size = max(1, shuffle_buffer_size)
        self.seed = seed
        self.epoch = 0
        self._num_examples = None

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        if self._num_examples is None:
            pf = pq.ParquetFile(self.filename)
            self._num_examples = pf.metadata.num_rows

        if self.world_size <= 1:
            return self._num_examples

        # Per-rank sample count after distributed sharding.
        base = self._num_examples // self.world_size
        rem = self._num_examples % self.world_size
        return base + (1 if self.rank < rem else 0)

    def _iter_rows(self):
        pf = pq.ParquetFile(self.filename)
        for batch in pf.iter_batches(batch_size=128):
            table = batch.to_pydict()
            for i in range(len(table["label_len"])):
                yield {
                    "token_ids": list(table["token_ids"][i]),
                    "case_labels": list(table["case_labels"][i]),
                    "punct_labels": list(table["punct_labels"][i]),
                    "valid_ids": list(table["valid_ids"][i]),
                    "token_masks": list(table["token_masks"][i]),
                    "label_masks": list(table["label_masks"][i]),
                    "label_len": table["label_len"][i],
                }

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_shards = self.world_size * num_workers
        shard_id = self.rank * num_workers + worker_id

        rng = random.Random(self.seed + self.epoch * 104729 + shard_id)
        buffer = []

        for rec_idx, row in enumerate(self._iter_rows()):
            if rec_idx % total_shards != shard_id:
                continue

            sample = (
                np.array(row["token_ids"]),
                np.array([row["case_labels"], row["punct_labels"]]),
                np.array(row["valid_ids"]),
                row["label_len"],
                np.array(row["label_masks"]),
            )

            if self.shuffle_buffer_size == 1:
                yield sample
                continue

            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(sample)
            else:
                pop_idx = rng.randrange(len(buffer))
                yield buffer.pop(pop_idx)
                buffer.append(sample)

        while buffer:
            pop_idx = rng.randrange(len(buffer))
            yield buffer.pop(pop_idx)


class DataModule(object):

    def __init__(self, args: argparse.Namespace, sp: sentencepiece):
        self.args = args
        self.sp = sp

        self.data_dir = self.args.data_dir

        self.train_features_file = f"{self.data_dir}/train_features.parquet"
        self.valid_features_file = f"{self.data_dir}/dev_features.parquet"
        self.test_features_file = f"{self.data_dir}/test_features.parquet"
        self.streaming_num_workers = getattr(self.args, "streaming_num_workers", 0)
        self.streaming_shuffle_buffer = getattr(self.args, "streaming_shuffle_buffer", 10000)

    def train_dataloader(self) -> DataLoader:
        logging.info(f"Using streaming parquet training dataset from {self.train_features_file}")
        train_dataset = StreamingParquetDataset(
            filename=self.train_features_file,
            max_seq_length=self.args.max_seq_length,
            world_size=self.args.world_size,
            rank=getattr(self.args, "rank", 0),
            shuffle_buffer_size=self.streaming_shuffle_buffer,
            seed=getattr(self.args, "seed", 42),
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.streaming_num_workers,
        )
        return train_dataloader

    def valid_dataloader(self) -> DataLoader:
        logging.info(f"Using streaming parquet validation dataset from {self.valid_features_file}")
        valid_dataset = StreamingParquetDataset(
            filename=self.valid_features_file,
            max_seq_length=self.args.max_seq_length,
            world_size=self.args.world_size,
            rank=getattr(self.args, "rank", 0),
            shuffle_buffer_size=1,
            seed=getattr(self.args, "seed", 42),
        )
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.streaming_num_workers,
        )
        return valid_dataloader

    def test_dataloader(self) -> DataLoader:
        logging.info(f"Using streaming parquet test dataset from {self.test_features_file}")
        test_dataset = StreamingParquetDataset(
            filename=self.test_features_file,
            max_seq_length=self.args.max_seq_length,
            world_size=self.args.world_size,
            rank=getattr(self.args, "rank", 0),
            shuffle_buffer_size=1,
            seed=getattr(self.args, "seed", 42),
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.streaming_num_workers,
        )
        return test_dataloader


def sort_batch(label_lens, valid_output, labels, label_masks, valid_ids=None):
    # print(f"before, label_lens:{label_lens}")
    label_lens, indx = label_lens.sort(dim=0, descending=True)

    valid_output = valid_output[indx]
    if labels is not None:
        labels = labels[indx]
    label_masks = label_masks[indx]
    if valid_ids is not None:
        valid_ids = valid_ids[indx]
        return label_lens, valid_output, labels, label_masks, valid_ids
    else:
        return label_lens, valid_output, labels, label_masks
