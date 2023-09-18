"""Data processing utilities."""

import json
import math
import random

import torch


def tab_printer(args):
    from texttable import Texttable

    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    datas = json.load(open(path))
    random.shuffle(datas)
    # print(f"file graphs: {len(datas)}\n")
    for data in datas:
        if len(data["graph_1"]) == 0 or len(data["graph_2"]) == 0:
            continue
        yield data


def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction - target) ** 2
    return score


def calculate_normalized_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    norm_ged = data["ged"] / (0.5 * (len(data["labels_1"]) + len(data["labels_2"])))
    return norm_ged


def assign_GPU(inputs, device):
    if type(inputs) == dict:
        output = {k: v.to(device) for k, v in inputs.items()}
    elif (type(inputs) == list) or (type(inputs) == tuple):
        output = [v.to(device) for v in inputs]
    else:
        output = inputs.to(device)
    return output


def euclidean_distance(x, y):
    """This is the squared Euclidean distance."""
    return torch.sum((x - y) ** 2, dim=-1)


def approximate_hamming_similarity(x, y):
    """Approximate Hamming similarity."""
    return torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)


def pairwise_loss(x, y, labels, loss_type="margin", margin=1.0):
    """Compute pairwise loss.

    Args:
      x: [N, D] float tensor, representations for N examples.
      y: [N, D] float tensor, representations for another N examples.
      labels: [N] int tensor, with values in -1 or +1.  labels[i] = +1 if x[i]
        and y[i] are similar, and -1 otherwise.
      loss_type: margin or hamming.
      margin: float scalar, margin for the margin loss.

    Returns:
      loss: [N] float tensor.  Loss for each pair of representations.
    """

    labels = labels.float()

    if loss_type == "margin":
        return torch.relu(margin - labels * (1 - euclidean_distance(x, y)))
    elif loss_type == "hamming":
        return 0.25 * (labels - approximate_hamming_similarity(x, y)) ** 2
    else:
        raise ValueError("Unknown loss_type %s" % loss_type)
