import os
import sys
import os.path as osp
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import datetime as dt

from torch_geometric.loader import NeighborLoader
from sklearn.metrics import average_precision_score

from models import RGIN, RGATGIN

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
parser.add_argument("--inference", type=bool, default=False)
parser.add_argument("--validation", type=bool, default=True)

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

logger = get_logger(log_dir)
logger.info(description)
logger.info(config)

# config
logger.info("config load success")

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("load data")
hgraph = torch.load(args.dataset)

labeled_class = args.labeled_class

if args.inference == False:
    train_idx = hgraph[labeled_class].pop("train_idx")
    if args.validation:
        val_idx = hgraph[labeled_class].pop("val_idx")
else:
    test_id = [int(x) for x in open(args.test_file).readlines()]
    converted_test_id = []
    for i in test_id:
        converted_test_id.append(hgraph["item"].maps[i])
    test_idx = torch.LongTensor(converted_test_id)

# Mini-Batch
if args.inference == False:
    train_loader = NeighborLoader(
        hgraph,
        input_nodes=(labeled_class, train_idx),
        num_neighbors=[config.fanout] * config.num_layers,
        shuffle=True,
        batch_size=config.batch_size,
    )

    if args.validation:
        val_loader = NeighborLoader(
            hgraph,
            input_nodes=(labeled_class, val_idx),
            num_neighbors=[config.fanout] * config.num_layers,
            shuffle=False,
            batch_size=config.batch_size,
        )
else:
    test_loader = NeighborLoader(
        hgraph,
        input_nodes=(labeled_class, test_idx),
        num_neighbors=[config.fanout] * config.num_layers,
        shuffle=False,
        batch_size=config.batch_size,
    )


num_relations = len(hgraph.edge_types)

if args.inference:
    model = torch.load(osp.join(checkpoint_dir, "best_model.pth"))
else:
    model = RGIN(
        in_channels=config.in_dim,
        hidden_channels=config.h_dim,
        out_channels=2,
        num_relations=num_relations,
        n_layers=config.num_layers,
        num_bases=config.num_bases,
        num_blocks=config.num_blocks,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-5)
logger.info(model)


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


crit = LabelSmoothing(0.2)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)), ascii=True)
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in train_loader:
        optimizer.zero_grad()
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
        loss = crit(y_hat, y)  # F.cross_entropy(y_hat, y)

        loss.backward()
        optimizer.step()
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
        loss = crit(y_hat, y)  # F.cross_entropy(y_hat, y)
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


def train_regression(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)), ascii=True)
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in train_loader:
        optimizer.zero_grad()
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
        loss = F.mse_loss(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch_size

        pbar.update(batch_size)
    pbar.close()
    return total_loss / total_examples


@torch.no_grad()
def val_regression():
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
        loss = F.mse_loss(y_hat, y)
        total_loss += float(loss) * batch_size
        pbar.update(batch_size)
    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test_regression():
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
        y_pred.append(y_hat.detach().cpu())
    pbar.close()

    return torch.hstack(y_pred)


if args.inference == False:
    logger.info("Start training")
    val_ap_list = []
    ave_val_ap = 0
    end = 0
    early_stopping = EarlyStopping(tolerance=3, best="max")
    best_metric = float("-inf")
    for epoch in range(1, config.n_epoch + 1):
        train_loss, train_acc, train_ap = train(epoch)
        logger.info(
            f"Train: Epoch {epoch:02d}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AP_Score: {train_ap:.4f}"
        )
        if args.validation and epoch >= config.early_stopping:
            val_loss, val_acc, val_ap = val()
            logger.info(
                f"Val: Epoch: {epoch:02d}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AP_Score: {val_ap:.4f}"
            )
            if val_ap > best_metric:
                logger.info("save model")
                torch.save(model, osp.join(checkpoint_dir, "best_model.pth"))

            # early stopping
            best_metric = max(best_metric, val_ap)
            logger.info("best metric: {}".format(best_metric))
            early_stopping(best_metric, monitor=val_ap)
            if early_stopping.early_stop:
                logger.info(f"We are at epoch: {epoch}")
                break

            val_ap_list.append(float(val_ap))
            ave_val_ap = np.average(val_ap_list)
            end = epoch
    logger.info(f"Complete Trianing")

#    with open(args.record_file, 'a+') as f:
#        f.write(f"{args.model_id:2d} {args.h_dim:3d} {args.n_layers:2d} {args.lr:.4f} {end:02d} {float(val_ap_list[-1]):.4f} {np.argmax(val_ap_list)+5:02d} {float(np.max(val_ap_list)):.4f}\n")


if args.inference == True:
    y_pred = test()
    with open(os.path.join(log_dir, config.ouput_result_file), "w+") as f:
        for i in range(len(test_id)):
            y_dict = {}
            y_dict["item_id"] = int(test_id[i])
            y_dict["score"] = float(y_pred[i])
            json.dump(y_dict, f)
            f.write("\n")
