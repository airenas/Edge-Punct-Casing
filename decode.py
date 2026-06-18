import argparse
import logging
import os
import random
from pathlib import Path
from typing import List, Dict

import sentencepiece as spm
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_module import DataModule
from egs.lt_ai_blkt.local.case import CASE_ID_MAP
from egs.lt_ai_blkt.local.punctuation import PUNCTUATION_ID_MAP
from train import get_model, get_params
from utils import (setup_logger)


# import onnxruntime as ort


##### usage
## python3 decode.py --data_dir ../data/ --exp_dir ../output/ --bpe_model ../bpe_model/bpe.model --batch 1000

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should include text file - words.txt and label file - labels.txt")
    parser.add_argument("--exp_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The experiment dir contains .pt")
    parser.add_argument("--bpe_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The bpe model path")
    parser.add_argument("--max_seq_length",
                        default=200,
                        type=int,
                        # required=True,
                        help="The sequence length of one sample after SentencePiece tokenization")
    parser.add_argument("--batch_size",
                        default=1024,
                        type=int,
                        # required=True,
                        help="Batch size for decoding")
    parser.add_argument("--world-size",
                        type=int,
                        default=1,
                        help="Number of GPUs for DDP training.", )
    parser.add_argument("--epoch",
                        default=-1,
                        type=int,
                        # required=True,
                        help="The epoch pt used for decoding")
    parser.add_argument("--batch",
                        default=-1,
                        type=int,
                        # required=True,
                        help="The batch pt used for decoding")
    parser.add_argument("--avg",
                        default=1,
                        type=int,
                        # required=True,
                        help="The number of checkpoints to average for decoding. Should be used together with --epoch or --batch")
    parser.add_argument("--file",
                        type=str,
                        required=True,
                        help="The test feature file to decode")
    return parser


def inc(d, k):
    if k in d:
        d[k] += 1
    else:
        d[k] = 1


def get_metrics(output, target):
    assert len(output) == len(target), f"output len:{output} != target len:{target}"

    true_predicted = {}
    all_predicted = {}
    all_expected = {}

    for i in range(len(output)):

        inc(all_expected, target[i])
        inc(all_predicted, output[i])
        if target[i] == output[i]:
            inc(true_predicted, output[i])

    # print(f"all_predicted:{all_predicted}")
    # print(f"all_expected:{all_expected}")
    # print(f"true_predicted:{true_predicted}")

    precision = {k: (true_predicted[k] if k in true_predicted else 0) / all_predicted[k] for k in all_predicted.keys()}
    recall = {k: (true_predicted[k] if k in true_predicted else 0) / all_expected[k] for k in all_expected.keys()}

    f_scores = {
        k: None if precision[k] == 0 else (
            0 if recall[k] == 0 else (2 * precision[k] * recall[k] / (precision[k] + recall[k])))
        for k in precision
    }

    overall_true_predicted = 0
    overall_all_predicted = 0
    overall_all_expected = 0
    for k in all_expected.keys():
        if k > 0:
            overall_true_predicted += (true_predicted[k] if k in true_predicted else 0)
            overall_all_predicted += (all_predicted[k] if k in all_predicted else 0)
            overall_all_expected += all_expected[k]
    overall_precision = (overall_true_predicted / overall_all_predicted if overall_all_predicted > 0 else 0)
    overall_recall = (overall_true_predicted / overall_all_expected if overall_all_expected > 0 else 0)
    overall_f_scores = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall) if overall_recall > 0 else 0)

    return precision, recall, f_scores, (overall_precision, overall_recall, overall_f_scores)


def get_counts(output, target):
    assert len(output) == len(target), f"output len:{output} != target len:{target}"

    predicted = {}
    expected = {}
    correct = {}

    for i in range(len(output)):
        inc(predicted, output[i])
        inc(expected, target[i])
        if output[i] == target[i]:
            inc(correct, output[i])

    return predicted, expected, correct


def print_metrics(logging, precision, recall, f_scores, overall, label_map):
    # print(f"precision:{precision}")

    for k in label_map.keys():
        # print(f"-----------> k:{k} - [{label_map[k]}]")
        logging.info(f"{label_map[k]}: \tPrec [{precision.get(k, 0):.3f}], " +
                     (f"\tRec [{recall[k]:.3f}], " if k in recall else "\tRec [None], ") +
                     (f"\tF1 [{f_scores[k]:.3f}], " if f_scores.get(k) != None else "\tF1 [None], ")
                     )
    logging.info(f"Overall: \tPrec [{overall[0]:.3f}], " +
                 f"\tRec [{overall[1]:.3f}], " +
                 f"\tF1 [{overall[2]:.3f}], "
                 )


def print_label_counts(logging, output, target, label_map, title):
    predicted, expected, correct = get_counts(output, target)
    total_expected = sum(expected.values())

    logging.info(title)
    for k in sorted(label_map.keys()):
        label_name = label_map[k]
        expected_pct = (100.0 * expected.get(k, 0) / total_expected) if total_expected > 0 else 0.0
        logging.info(
            f"{k} -> {label_name}: predicted [{predicted.get(k, 0)}], expected [{expected.get(k, 0)}] ({expected_pct:.2f}%), correct [{correct.get(k, 0)}]"
        )


def average_checkpoints(
        filenames: List[Path], device: torch.device = torch.device("cpu")
) -> dict:
    """Average a list of checkpoints.

    Args:
      filenames:
        Filenames of the checkpoints to be averaged. We assume all
        checkpoints are saved by :func:`save_checkpoint`.
      device:
        Move checkpoints to this device before averaging.
    Returns:
      Return a dict (i.e., state_dict) which is the average of all
      model state dicts contained in the checkpoints.
    """
    n = len(filenames)
    for filename in filenames:
        logging.info(f"Loading checkpoint from {filename}")

    avg = torch.load(filenames[0], map_location=device, weights_only=False)["model"]

    # Identify shared parameters. Two parameters are said to be shared
    # if they have the same data_ptr
    uniqued: Dict[int, str] = dict()

    for k, v in avg.items():
        v_data_ptr = v.data_ptr()
        if v_data_ptr in uniqued:
            continue
        uniqued[v_data_ptr] = k

    uniqued_names = list(uniqued.values())

    for i in range(1, n):
        state_dict = torch.load(filenames[i], map_location=device, weights_only=False)[
            "model"
        ]
        for k in uniqued_names:
            avg[k] += state_dict[k]

    for k in uniqued_names:
        if avg[k].is_floating_point():
            avg[k] /= n
        else:
            avg[k] //= n

    return avg


@torch.no_grad()
def main():
    parser = get_parser()

    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    params = get_params()
    params.update(vars(args))

    random.seed(42)
    torch.manual_seed(42)

    setup_logger(f"{params.exp_dir}/log-decode", use_console=False)
    logging.info("Decoding started")

    device = torch.device("cpu")
    rank = 0  # hardcode 0 to use single GPU firstly
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)
    print(model)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    files = []
    if params.epoch > 0:
        for i in range(params.avg):
            files.append(f"{params.exp_dir}/epoch-{params.epoch - i}.pt")
    if params.batch > 0:
        files = []
        for i in range(params.avg):
            files.append(f"{params.exp_dir}/checkpoint-{params.batch - (i * 1000)}.pt")
    # logging.info(f"Loading checkpoint from {ptfile}")
    # checkpoint = torch.load(ptfile, map_location="cpu")
    # checkpoint.pop("model")
    # model.load_state_dict(checkpoint["model"], strict=False)
    files.sort()
    model.load_state_dict(average_checkpoints([Path(f) for f in files]), strict=False)

    model.to(device)
    model.eval()

    data_module = DataModule(args, sp)
    decode_dl = data_module.test_dataloader(file=args.file)
    logging.info(f"len(decode_dl):{len(decode_dl)}")

    all_case_pred = []
    all_case_labels = []

    all_punct_pred = []
    all_punct_labels = []

    for batch_idx, batch in enumerate(tqdm(decode_dl)):
        batch = tuple(t.to(device) for t in batch)
        token_ids, label_ids, valid_ids, label_lens, label_masks = batch

        active_case_logits, active_punct_logits, mask, indx = model(token_ids, valid_ids=valid_ids,
                                                                    label_lens=label_lens)

        # label_lens, indx = torch.sort(label_lens, dim=0, descending=True, stable=True)
        label_ids = label_ids[indx]

        case_pred = torch.argmax(F.log_softmax(active_case_logits, dim=1), dim=1)
        punct_pred = torch.argmax(F.log_softmax(active_punct_logits, dim=1), dim=1)

        label_ids = label_ids[:, :, :mask.shape[1]]
        active_case_labels = label_ids[:, 0, :][mask]
        active_punct_labels = label_ids[:, 1, :][mask]

        all_case_pred.extend(case_pred.detach().cpu().tolist())
        all_case_labels.extend(active_case_labels.detach().cpu().tolist())
        all_punct_pred.extend(punct_pred.detach().cpu().tolist())
        all_punct_labels.extend(active_punct_labels.detach().cpu().tolist())

    total_precision_case, total_recall_case, total_f_scores_case, total_overall_case = get_metrics(
        all_case_pred, all_case_labels
    )
    total_precision_punct, total_recall_punct, total_f_scores_punct, total_overall_punct = get_metrics(
        all_punct_pred, all_punct_labels
    )

    logging.info(
        "\nCase metrics:\n----------------------------------------------------------------------------------------")
    print_metrics(logging, total_precision_case, total_recall_case, total_f_scores_case, total_overall_case,
                  CASE_ID_MAP)
    logging.info("\n")
    print_label_counts(logging, all_case_pred, all_case_labels, CASE_ID_MAP, "Case label counts")
    logging.info(
        "\nPunct metrics:\n=======================================================================================")
    print_metrics(logging, total_precision_punct, total_recall_punct, total_f_scores_punct, total_overall_punct,
                  PUNCTUATION_ID_MAP)
    logging.info("\n")
    print_label_counts(logging, all_punct_pred, all_punct_labels, PUNCTUATION_ID_MAP, "Punctuation label counts")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        format=formatter,
        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING),
    )

    logging.info("Starting")
    main()
    logging.info("Done")
