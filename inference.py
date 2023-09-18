import argparse
import os
import pickle
import time
import csv
import json
import datetime as dt
import pathlib

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from datasets import TestDataset, test_collate
from models import BinaryModel
from utils import ConfigParse, get_logger


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
    header = "fid,fid0:sim0,fid1:sim1,fid2:sim2,fid3:sim3,fid4:sim4,fid5:sim5,fid6:sim6,fid7:sim7,fid8:sim8,fid9:sim9".split(
        ","
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
    parser = argparse.ArgumentParser(description="test model")
    parser.add_argument(
        "-c",
        "--config_file",
        help="Path to .json file of training parameters",
        default="config.json",
    )
    parser.add_argument(
        "-m",
        "--model_file",
        help="Path to model file",
        default="./logs/20221009-15:49/model.pt",
    )
    parser.add_argument(
        "--comment",
        type=str,
        help="comment your job",
        default="inference",
    )
    args = parser.parse_args()

    config_file = json.load(open(args.config_file, "r"))
    config = ConfigParse(config_file)

    strtime = dt.datetime.now().strftime("%Y%m%d-%H:%M")
    checkpoint_path = f"./logs/{strtime}"
    logger = get_logger(checkpoint_path)

    logger.info(config_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_file = args.model_file
    logger.info(args.model_file)

    model = BinaryModel(args=config).to(device)
    model.load_state_dict(torch.load(model_file))

    inference()
