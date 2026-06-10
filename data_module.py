import argparse
import logging
import os

import numpy as np
import sentencepiece
from torch.utils.data import (Dataset, DataLoader, RandomSampler)
from torch.utils.data.distributed import DistributedSampler
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


class DataModule(object):

    def __init__(self, args: argparse.Namespace, sp: sentencepiece):
        self.args = args
        self.sp = sp

        self.data_dir = self.args.data_dir

        self.train_dataset = TextDataset()
        self.valid_dataset = TextDataset()
        self.test_dataset = TextDataset()

        self.train_features_file = f"{self.data_dir}/train_features.txt"
        self.valid_features_file = f"{self.data_dir}/dev_features.txt"
        self.test_features_file = f"{self.data_dir}/test_features.txt"

    def train_dataloader(self) -> DataLoader:
        self.train_dataset.load_features(self.train_features_file, self.args.max_seq_length)
        if self.args.world_size > 1:
            train_sampler = DistributedSampler(self.train_dataset)
        # shuffle = False
        else:
            train_sampler = RandomSampler(self.train_dataset)
        # shuffle = True
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.batch_size,
            # shuffle=shuffle,
        )

        return train_dataloader

    def valid_dataloader(self) -> DataLoader:
        self.valid_dataset.load_features(self.valid_features_file, self.args.max_seq_length)

        if self.args.world_size > 1:
            valid_sampler = DistributedSampler(self.valid_dataset)
        else:
            valid_sampler = RandomSampler(self.valid_dataset)
        valid_dataloader = DataLoader(
            dataset=self.valid_dataset,
            sampler=valid_sampler,
            batch_size=self.args.batch_size
        )

        return valid_dataloader

    def test_dataloader(self) -> DataLoader:
        if not os.path.isfile(self.test_features_file):
            logging.info("Extracting test features:")
            self.test_dataset.convert_examples_to_features_bos_eos(self.args.max_seq_length, self.sp)

            logging.info("First time to extract features, save features to local file for next time quick load...")
            self.test_dataset.save_features(self.test_features_file)
        else:
            logging.info("Test feature file already exists, loading...")
            self.test_dataset.load_features(self.test_features_file, self.args.max_seq_length)
        # print(f"print first 2 example in test dataset:\n{self.test_dataset[:2]}")

        if self.args.world_size > 1:
            test_sampler = DistributedSampler(self.test_dataset)
        else:
            test_sampler = RandomSampler(self.test_dataset)
        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            sampler=test_sampler,
            batch_size=self.args.batch_size
        )

        return test_dataloader, self.test_text


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
