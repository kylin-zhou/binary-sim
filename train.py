import argparse
import datetime as dt
import json
import os
import csv
import pickle
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from datasets.graph_dataset import GraphEmbeddingDataset
from models import GIN
from utils import ConfigParse, get_logger, sup_loss, unsup_loss


class Args:
    seed = 2020
    epochs = 10000
    node_feature_dim = 64
    nhid = 64
    batch_size = 256
    dropout = 0.1
    learning_rate = 0.001
    ratio1 = 1.0
    ratio2 = 1.0
    ratio3 = 0.8
    weight_decay = 5e-4
    device = "cuda:0"
    mode = "RW"  # Specify hypergraph construction mode NEighbor(NE)/RandomWalk(RW)
    patience = 3
    k = 5  # Hyperparameter for construction hyperedge


args = Args()
dataset = GraphEmbeddingDataset(args)
args.num_features = dataset.number_features

device = args.device
model = GIN(args).to(args.device)
# model = BaseModel(args).to(args.device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
)

loss_fn = unsup_loss


def train():
    print("\nModel training.\n")
    start = time.time()
    val_loss_values = []
    patience_cnt = 0
    best_epoch = 0
    min_loss = 1e10

    # with torch.autograd.detect_anomaly():
    for epoch in range(args.epochs):
        model.train()
        all_loss = []
        batches = dataset.create_batches(dataset.training_funcs, dataset.collate)
        for index, batch_pair in enumerate(batches):
            optimizer.zero_grad()
            # print(data["g1"])
            pred = model(batch_pair.to(device))
            loss = loss_fn(pred, device=device)
            loss.backward()
            optimizer.step()
            all_loss.append(loss.item())
        loss = sum(all_loss) / len(all_loss)
        # start validate at 9000th iteration
        if epoch + 1 < 5:
            end = time.time()
            print(
                "Epoch: {:05d},".format(epoch + 1),
                "loss_train: {:.6f},".format(loss),
                "time: {:.6f}s".format(end - start),
            )
        else:
            val_loss, aucscore = validate(dataset, dataset.validation_funcs)
            end = time.time()
            print(
                "Epoch: {:05d},".format(epoch + 1),
                "loss_train: {:.6f},".format(loss),
                "loss_val: {:.6f},".format(val_loss),
                "AUC: {:.6f},".format(aucscore),
                "time: {:.6f}s".format(end - start),
            )
            val_loss_values.append(val_loss)
            if val_loss_values[-1] < min_loss:
                min_loss = val_loss_values[-1]
                patience_cnt = 0
                torch.save(
                    model.state_dict(),
                    "checkpoint/graph_model.pth",
                )
            else:
                patience_cnt += 1

            if patience_cnt == args.patience:
                print(f"early stopping in epoch {epoch}")
                break

    print(
        "Optimization Finished! Total time elapsed: {:.6f}".format(time.time() - start)
    )


def validate(datasets, funcs):
    model.eval()
    all_loss = []
    with torch.no_grad():
        batches = datasets.create_batches(funcs, datasets.collate)
        for index, batch_pair in enumerate(batches):
            pred = model(batch_pair.to(device))
            loss = loss_fn(pred, device=device)
            all_loss.append(loss.item())

        loss = sum(all_loss) / len(all_loss)

        return loss, 0


if __name__ == "__main__":
    train()
    best_model = "checkpoint/graph_model.pth"
    model.load_state_dict(torch.load("{}.pth".format(best_model)))
    print("\nModel evaluation.")
    test_loss, test_auc = validate(dataset, dataset.testing_funcs)
    print("Test set results, loss = {:.6f}, AUC = {:.6f}".format(test_loss, test_auc))
