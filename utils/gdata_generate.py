"""
{"graph_1": [[0, 1], [1, 2], [2, 3], [3, 4]],
 "graph_2":  [[0, 1], [1, 2], [1, 3], [3, 4], [2, 4]],
 "labels_1": [2, 2, 2, 2],
 "labels_2": [2, 3, 2, 2, 2],
 "ged": 1}
"""

import binascii
import csv
import json
import random
from itertools import combinations

import networkx as nx
import numpy as np
from tqdm import tqdm
import pickle

# convert cfg/node_label/pair_label to json
# train/test set

print("read func")
func_file = "data/public/train.func.json"
with open(func_file, "r") as f:
    lines = f.readlines()
func_maps = {}
for line in tqdm(lines):
    pline = json.loads(line)
    func_maps[str(pline["fid"])] = pline["cfg"]

fids = list(func_maps.keys())

print("genrate pair data")
file = "data/public/train.group.csv"
index, max_index, train_valid_index = 0, 1000, 900
with open(file, "r") as f:
    reader = csv.reader(f)
    for row in tqdm(reader):
        group_id = row[0]
        if index < train_valid_index:
            train_valid = "train"
        else:
            train_valid = "valid"
        f = open(
            f"data/{train_valid}/{group_id}.json", "w"
        )
        combines = list(combinations(row[1:], 2))
        pair_data = []
        for pair in combines:
            g1, g2 = func_maps[pair[0]], func_maps[pair[1]]
            data = {
                "graph_1": g1,
                "graph_2": g2,
                "ged": 1,
            }
            pair_data.append(data)

            neg_g = func_maps[random.sample(fids, 1)[0]]
            data = {
                "graph_1": g1,
                "graph_2": neg_g,
                "ged": 0,
            }
            pair_data.append(data)

        json.dump(pair_data, f)
        f.close()

        if index > max_index:
            break
        index += 1
