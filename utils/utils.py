import os
import logging
import datetime as dt

import torch
import torch.nn as nn
import torch.nn.functional as F


def unsup_loss(y_pred, lamda=0.05, device="cpu"):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)

    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12

    similarities = similarities / lamda

    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def sup_loss(y_pred, lamda=0.05, device="cpu"):
    row = torch.arange(0, y_pred.shape[0], 3, device=device)
    col = torch.arange(y_pred.shape[0], device=device)
    col = torch.where(col % 3 != 0)[0]
    y_true = torch.arange(0, len(col), 2, device=device)
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)

    similarities = torch.index_select(similarities, 0, row)
    similarities = torch.index_select(similarities, 1, col)

    similarities = similarities / lamda

    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def get_logger(output_path=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Create a standard formatter
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    # Create a handler for output to the console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create a handler for output to logfile
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        log_file = os.path.join(output_path, "log.txt")
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


from collections import namedtuple


class ConfigParse(object):
    """convert dict to object
    following resources: https://www.cjavapy.com/article/2590/
    """

    def __new__(cls, data):
        if isinstance(data, dict):
            return namedtuple("struct", data.keys())(*(cls(val) for val in data.values()))
        elif isinstance(data, (tuple, list, set, frozenset)):
            return type(data)(cls(_) for _ in data)
        else:
            return data


def average_precision(y_true, pred):
    """Computes the average precision.
    gt: list, ground truth, all relevant docs' index
    pred: list, prediction
    """
    if not y_true:
        return 0.0

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(pred):
        if p in y_true and p not in pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / max(1.0, len(y_true))


def mean_average_precision(y_true, pred):
    """
    y_true: {q:[a1,a2,……]}
    pred: {q:[a1,a2,……]}
    """
    ap_list = []
    for i in range(len(y_true)):
        ap = average_precision(y_true[i], pred[i])
    return sum(ap_list) / max(1.0, len(ap_list))


def NDCG(y_true, pred, use_graded_scores=False):
    """Computes the NDCG.
    y_true: list, ground truth, all relevant docs' index
    pred: list, prediction
    """
    score = 0.0
    for rank, item in enumerate(pred):
        if item in y_true:
            if use_graded_scores:
                grade = 1.0 / (y_true.index(item) + 1)
            else:
                grade = 1.0
            score += grade / np.log2(rank + 2)

    norm = 0.0
    for rank in range(len(y_true)):
        if use_graded_scores:
            grade = 1.0 / (rank + 1)
        else:
            grade = 1.0
        norm += grade / np.log2(rank + 2)
    return score / max(0.3, norm)
