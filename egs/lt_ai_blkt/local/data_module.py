import logging
from dataclasses import dataclass

from tqdm import tqdm

from egs.lt_ai_blkt.local.case import get_case_id
from egs.lt_ai_blkt.local.parquet_utils import FeatureParquetKeeper
from egs.lt_ai_blkt.local.punctuation import get_punctuation_id
from egs.lt_ai_blkt.local.utils import split_word_punctuation


@dataclass
class InputFeatures:
    token_ids: list
    label_ids: list
    valid_ids: list
    token_masks: list
    label_masks: list
    label_len: int


def split_text(line):
    words = []
    case_labels = []
    punct_labels = []
    for wi in line.split():
        w, p = split_word_punctuation(wi)
        words.append(w.lower())
        case_labels.append(get_case_id(w))
        punct_labels.append(get_punctuation_id(p))
    return words, case_labels, punct_labels


class TextDatasetNew:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_tokens_num(self, words):
        num = 0
        for word in words:
            num += len(self.tokenizer.encode(word, out_type=int))
        return num

    def _iter_examples_stream(self, text_fp):
        for line in text_fp:
            line = line.strip()
            if not line:
                continue
            words, case_labels, punct_labels = split_text(line)
            yield words, case_labels, punct_labels, line

    def _build_padded_feature(self, tokens, labels, valid, max_seq_length, eos_id):
        tokens = list(tokens)
        labels = [list(labels[0]), list(labels[1])]
        valid = list(valid)

        tokens.append(eos_id)
        labels[0].append(0)
        labels[1].append(0)
        valid.append(1)

        token_masks = [1] * len(tokens)
        label_masks = [1] * len(labels[0])
        label_len = len(labels[0])

        while len(tokens) < max_seq_length:
            tokens.append(0)
            token_masks.append(0)
            valid.append(0)

        while len(labels[0]) < max_seq_length:
            labels[0].append(0)
            labels[1].append(0)
            label_masks.append(0)

        return InputFeatures(
            token_ids=tokens,
            label_ids=labels,
            valid_ids=valid,
            token_masks=token_masks,
            label_masks=label_masks,
            label_len=label_len,
        )

    def iter_features_bos_eos(self, max_seq_length, in_fp):
        bos_id = self.tokenizer.piece_to_id("<s>")
        eos_id = self.tokenizer.piece_to_id("</s>")

        tokens = [bos_id]
        labels = [[0], [0]]
        valid = [1]
        last_tokens = []
        last_labels = [[], []]
        last_valid = []

        logging.info(
            "Converting examples to features... bos_id:%s, eos_id:%s", bos_id, eos_id
        )

        for words, case_labels, punct_labels, raw_line in self._iter_examples_stream(in_fp):
            tokens_num = self.get_tokens_num(words)
            if tokens_num >= max_seq_length - 10:
                logging.info("tokens num:[%s] ----> %s", tokens_num, raw_line.rstrip("\n"))
                continue

            for iw, word in enumerate(words):
                word_tokens = self.tokenizer.encode(word, out_type=int)

                if len(tokens) + len(word_tokens) > max_seq_length - 1:
                    yield self._build_padded_feature(tokens, labels, valid, max_seq_length, eos_id)

                    tokens = [bos_id]
                    labels = [[0], [0]]
                    valid = [1]

                    if iw > 0:
                        tokens.extend(last_tokens)
                        labels[0].extend(last_labels[0])
                        labels[1].extend(last_labels[1])
                        valid.extend(last_valid)

                if iw == 0:
                    last_tokens = []
                    last_labels = [[], []]
                    last_valid = []

                tokens.extend(word_tokens)
                last_tokens.extend(word_tokens)

                for m in range(len(word_tokens)):
                    if m == 0:
                        labels[0].append(case_labels[iw])
                        labels[1].append(punct_labels[iw])
                        valid.append(1)

                        last_labels[0].append(case_labels[iw])
                        last_labels[1].append(punct_labels[iw])
                        last_valid.append(1)
                    else:
                        valid.append(0)
                        last_valid.append(0)

    def convert_examples_to_features(
            self, max_seq_length, input_file, output_file
    ):
        written = 0
        with FeatureParquetKeeper(output_file, max_seq_length=max_seq_length) as out_keeper:
            with open(input_file, "r", encoding="utf-8") as in_fp:
                for feature in tqdm(self.iter_features_bos_eos(max_seq_length, in_fp)):
                    out_keeper.feed_feature(feature)
                    written += 1

        logging.info("Saved %s features to %s", written, output_file)
