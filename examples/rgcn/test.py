import os
import sys
import os.path as osp
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import datetime as dt

from torch_geometric.loader import NeighborLoader
from sklearn.metrics import average_precision_score

from models import RGIN


from utils import (
    ConfigParse,
    EarlyStopping,
    create_explog,
    file_path,
    get_logger,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config_file",
    default="config.json",
    help="Path to .json file of training parameters",
)
parser.add_argument("--dataset", type=str, default="/home/icdm/pyg_data/icdm2022.pt")
parser.add_argument(
    "--test-file", type=str, default="/data1/test/data/test_session1_ids.csv"
)
parser.add_argument("--labeled-class", type=str, default="item")
parser.add_argument("--device-id", type=str, default="0")
parser.add_argument("--validation", type=bool, default=False)
parser.add_argument("--inference", type=bool, default=False)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="logs/icdm/rgcn/202208252033/checkpoint",
)

args = parser.parse_args()

# comment: Description of this experiment
description = "rgcn-gin"

# load config
config = ConfigParse(json.load(open(args.config_file, "r")))

# get logger
workdir = ""
strtime = dt.datetime.now().strftime("%Y%m%d%H%M")
log_dir, checkpoint_dir = create_explog(
    workdir, strtime, args.config_file, name="logs/icdm/rgcn"
)
if args.checkpoint_dir:
    checkpoint_dir = args.checkpoint_dir

logger = get_logger(log_dir)
logger.info(description)
logger.info(config)

# config
logger.info("config load success")

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


labeled_class = args.labeled_class

# Mini-Batch
if args.validation:
    logger.info("validation")
    logger.info("load data")
    hgraph = torch.load(args.dataset)

    val_idx = hgraph[labeled_class].pop("val_idx")

    val_loader = NeighborLoader(
        hgraph,
        input_nodes=(labeled_class, val_idx),
        num_neighbors=[config.fanout] * config.num_layers,
        shuffle=False,
        batch_size=config.batch_size,
    )

if args.inference:
    logger.info("inference")
    logger.info("load data")
    hgraph = torch.load(args.dataset)

    test_id = [int(x) for x in open(args.test_file).readlines()]
    converted_test_id = []
    for i in test_id:
        converted_test_id.append(hgraph["item"].maps[i])
    test_idx = torch.LongTensor(converted_test_id)

    test_loader = NeighborLoader(
        hgraph,
        input_nodes=(labeled_class, test_idx),
        num_neighbors=[config.fanout] * config.num_layers,
        shuffle=False,
        batch_size=config.batch_size,
    )


num_relations = len(hgraph.edge_types)

model = torch.load(osp.join(checkpoint_dir, "best_model.pth"))
logger.info(model)


@torch.no_grad()
def val():
    model.eval()
    pbar = tqdm(total=int(len(val_loader.dataset)), ascii=True)
    pbar.set_description("val")
    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in val_loader:
        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)
        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()

        y_hat = model(
            batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device)
        )[start : start + batch_size]
        loss = F.cross_entropy(y_hat, y)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += float(loss) * batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch_size
        pbar.update(batch_size)
    pbar.close()
    ap_score = average_precision_score(
        torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy()
    )

    return total_loss / total_examples, total_correct / total_examples, ap_score


@torch.no_grad()
def test():
    model.eval()
    pbar = tqdm(total=int(len(test_loader.dataset)), ascii=True)
    pbar.set_description(f"Generate Final Result:")
    y_pred = []
    for batch in test_loader:
        batch_size = batch[labeled_class].batch_size
        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()
        y_hat = model(
            batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device)
        )[start : start + batch_size]
        pbar.update(batch_size)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
    pbar.close()

    return torch.hstack(y_pred)


if args.validation:
    val_loss, val_acc, val_ap = val()
    logger.info(
        f"Val: Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AP_Score: {val_ap:.4f}"
    )


if args.inference == True:
    y_pred = test()
    with open(os.path.join(log_dir, config.ouput_result_file), "w+") as f:
        for i in range(len(test_id)):
            y_dict = {}
            y_dict["item_id"] = int(test_id[i])
            y_dict["score"] = float(y_pred[i])
            json.dump(y_dict, f)
            f.write("\n")
