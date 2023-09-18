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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from datasets import (
    PairDataset,
    TripletDataset,
    collate,
    test_collate,
    TestDataset,
)
from models import BinaryModel
from utils import ConfigParse, get_logger, sup_loss, unsup_loss


def process_batch(batch):
    inputs = {}
    if config.bytecode:
        inputs["code"] = batch["code"].to(device)
    if config.asm:
        inputs["asm"] = batch["asm"].to(device)
    if config.graph:
        inputs["graph"] = batch["graph"].to(device)
    if config.wide:
        inputs["wide"] = batch["wide"].to(device)
    if config.integer:
        inputs["integer"] = batch["integer"].to(device)
    if config.image:
        inputs["image"] = batch["image"].to(device)
    return inputs


def train(dataloader, optimizer, scheduler, loss_fn):
    model.train()
    all_loss = []

    pbar = tqdm(total=int(len(dataloader.dataset) * range_base), ascii=True)
    pbar.set_description("train")
    for idx, batch in enumerate(dataloader):
        inputs = process_batch(batch)

        pred = model(inputs)
        optimizer.zero_grad()
        loss = loss_fn(pred, device=device)
        loss.backward()
        optimizer.step()

        all_loss.append(loss.item())
        pbar.update(len(pred))
    pbar.close()

    scheduler.step()

    return sum(all_loss) / len(all_loss)


@torch.no_grad()
def valid(dataloader, loss_fn):
    model.eval()
    all_loss = []

    pbar = tqdm(total=int(len(dataloader.dataset) * range_base), ascii=True)
    pbar.set_description("val")
    for i, batch in enumerate(dataloader):
        inputs = process_batch(batch)

        pred = model(inputs)
        loss = loss_fn(pred, device=device)
        all_loss.append(loss.item())
        pbar.update(len(pred))
    pbar.close()

    return sum(all_loss) / len(all_loss)


def inference():
    dataset = TestDataset(config)
    test_dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=test_collate,
    )

    logger.info("start inference")

    start = time.time()

    # inference in test
    look_up = {}
    with torch.no_grad():
        model.eval()
        for i, batch in tqdm(enumerate(test_dataloader)):
            fids = batch["fid"]
            inputs = process_batch(batch)

            pred = model(inputs)
            batch_dict = {str(k): v for k, v in zip(fids, pred)}
            look_up.update(batch_dict)

    logger.info(
        "time: {:.6f}s".format(time.time() - start),
    )

    # read question file
    querys = []
    candidates = []
    with open(config.question_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            querys.append(row[0])
            candidates.append(row[1:])

    # write to result csv
    logger.info("write to result csv")
    result_file = os.path.join(checkpoint_path, "result.csv")
    if os.path.exists(result_file):
        os.remove(result_file)
    f = open(result_file, "a")
    writer = csv.writer(f)
    header = (
        "fid,fid0:sim0,fid1:sim1,fid2:sim2,fid3:sim3,fid4:sim4,fid5:sim5,fid6:sim6,fid7:sim7,fid8:sim8,fid9:sim9".split(
            ","
        )
    )
    writer.writerow(header)

    # get_query_result
    for query, candidate in tqdm(zip(querys, candidates)):
        vec = []
        for a in candidate:
            vec.append(look_up.get(a, torch.zeros((1, 64), device=device)))
        answer_vecs = torch.vstack(vec)
        query_vec = look_up.get(query, torch.zeros((1, 64), device=device))
        sim = F.cosine_similarity(query_vec, answer_vecs)

        indexs = torch.sort(sim, descending=True)[1][:10]
        res = [query] + [f"{candidate[i]}:{sim[i]}" for i in indexs]

        writer.writerow(res)

    f.close()

    logger.info("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "-c",
        "--config_file",
        help="Path to .json file of training parameters",
        default="config.json",
    )
    parser.add_argument(
        "--comment",
        type=str,
        help="comment your job",
        default="train binary sim, bytecode+asm+cfg+integer",
    )
    args = parser.parse_args()

    config_file = json.load(open(args.config_file, "r"))
    config = ConfigParse(config_file)

    strtime = dt.datetime.now().strftime("%Y%m%d-%H:%M")
    checkpoint_path = f"./logs/{strtime}"
    logger = get_logger(checkpoint_path)

    logger.info(args.comment)
    logger.info(config_file)

    task = config.task
    if task == "pair":
        dataset = PairDataset(config)
        loss_fn = unsup_loss
        range_base = 2
    elif task == "triplet":
        dataset = TripletDataset(config)
        loss_fn = sup_loss
        range_base = 3
    assert task in ("pair", "triplet")

    logger.info(f"task: {task}")
    logger.info("len dataset: {}".format(len(dataset)))

    # train/valid
    valid_index = int(len(dataset) * 0.9)
    train_set, valid_set = random_split(dataset, [valid_index, len(dataset) - valid_index])
    train_dataloader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=collate,
    )
    valid_dataloader = DataLoader(
        valid_set,
        batch_size=config.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=collate,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BinaryModel(args=config).to(device)
    logger.info(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)

    logger.info("start training")
    best_loss = 1e10
    for epoch in range(config.epochs):
        start = time.time()
        loss = train(train_dataloader, optimizer, scheduler, loss_fn=loss_fn)

        if epoch < config.early_stopping:
            logger.info(
                "Epoch: {}, loss_train: {:.6f}, time: {:.6f}s".format(epoch, loss, time.time() - start),
            )
        else:
            val_loss = valid(valid_dataloader, loss_fn=loss_fn)

            logger.info(
                "Epoch: {}, loss_train: {:.6f}, loss_val: {:.6f}, time: {:.6f}s".format(
                    epoch, loss, val_loss, time.time() - start
                ),
            )

            if val_loss < best_loss:
                best_loss = val_loss
                count = 0
                torch.save(model.state_dict(), f"{checkpoint_path}/model.pt")
            else:
                count += 1
                if count >= config.patience:
                    logger.info(f"Finished! early stop at epoch {epoch}")
                    break

    # inference
    inference()
