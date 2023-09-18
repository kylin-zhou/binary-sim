import csv
import json
import time
from collections.abc import Iterable

import numpy as np
import torch
from tqdm import tqdm

from param_parser import parameter_parser
from simgnn import GraphEmbedding, SimGNN
from utils import assign_GPU


# cosine sim
def pairwise_dot_product_similarity(x, y):
    # x.shape [1,dim], y.shape [n,dim]
    # dot = x * y
    # factor = np.linalg.norm(x) * np.linalg.norm(y, axis=1) + 1
    # return np.sum(x * y, axis=1) / factor
    num = np.dot(x, np.array(y).T)  # 向量点乘
    denom = np.linalg.norm(x) * np.linalg.norm(y, axis=1)  # 求模长的乘积
    res = num / denom
    return res


def euclidean_similarity(x, y):
    """This is the squared Euclidean distance."""
    scores = -torch.sum((x - y) ** 2, dim=-1)
    scores_max = torch.max(scores)
    scores_min = torch.min(scores)

    # normalize scores to [0, 1] and add a small epislon for safety
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)
    return scores


def get_test_func():
    func_file = "datadata/public/test.func.json"
    with open(func_file, "r") as f:
        lines = f.readlines()

    test_func_maps = {}
    for line in tqdm(lines):
        pline = json.loads(line)
        cfg = pline["cfg"]
        if len(cfg) > 1:
            test_func_maps[pline["fid"]] = cfg
    return test_func_maps


def get_query_candidate():
    file = "datadata/public/test.question.csv"

    querys = []
    candidates = []
    with open(file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            querys.append(int(row[0]))
            candidates.append([int(i) for i in row[1:]])
    return querys, candidates


def process_cfg(cfg):
    """cfg to tensor
    Transferring the data to torch and creating a hash table.
    Including the indices, features and target.
    :param data: CFG Data.
    :return new_data: Torch Tensors.
    """
    edges = cfg
    edges = np.array(edges, dtype=np.int64)

    num_of_nodes = edges.max() + 1
    node_feature_dim = 32

    edges = torch.from_numpy(edges.T).type(torch.long)

    features = np.random.randn(num_of_nodes, node_feature_dim)
    features = torch.FloatTensor(np.array(features))

    return edges, features


def prediction(model, x):
    """
    args = parameter_parser()

    model_file = "checkpoint/simgnn_model.pt"
    model = SimGNN(args=args, number_of_labels=32)
    model.load_state_dict(torch.load(model_file))
    x = (
        torch.tensor(np.random.randint(1, 10, (2, 10)), dtype=torch.long, device=device),
        torch.tensor(np.random.randn(10, 32), dtype=torch.float32, device=device),
    )
    return:
        node embedding:
        graph embedding:
    """
    out = model.graph_embedding(x)
    # print(len(out), out[0].shape, out[1].shape, out[1].reshape(1, -1).shape)
    return out[1].detach().cpu().numpy().reshape(1, -1)[0]


def get_query_result(query, candidate, look_up):
    vec = []
    for a in candidate:
        vec.append(look_up.get(a, np.zeros(32)))
    answer_vecs = np.stack(vec, axis=0)
    query_vec = look_up.get(query, np.ones(32))
    sim = pairwise_dot_product_similarity(query_vec, answer_vecs)

    index = sorted(
        [[i, v] for i, v in enumerate(sim)], key=lambda x: x[1], reverse=True
    )[:10]
    res = str(q) + "," + ",".join([f"{candidate[i[0]]}:{i[1]}" for i in index])
    return res


def group_test():
    print("process function data")
    with open("group0.json", "r") as f:
        lines0 = json.load(f)
    with open("group3.json", "r") as f:
        lines3 = json.load(f)
    lines = lines0 + lines3
    test_func_maps = {}
    for line in tqdm(lines):
        cfg = line["cfg"]
        if len(cfg) > 1:
            test_func_maps[str(line["fid"])] = cfg

    vecs = []
    for k, v in tqdm(test_func_maps.items()):
        x = process_cfg(v)
        pred = prediction(model, x)
        vecs.append(pred)
    vecs = np.stack(vecs, axis=0)
    for i in range(10):
        x = vecs[i].reshape(1, -1)
        y = vecs
        print(euclidean_similarity(torch.tensor(x), torch.tensor(y)))


device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
# load model
args = parameter_parser()
model_file = "checkpoint/simgnn_model.pt"
model = SimGNN(args=args, node_feature_dim=32).to(device)
model.load_state_dict(torch.load(model_file))

# prediction(model, x)
group_test()

# # get graph embedding
# print("get func")
# func_maps = get_test_func()
# print("get graph embdding")
# graph_vec_maps = {}
# for k, v in tqdm(func_maps.items()):
#     x = process_cfg(v)
#     x = assign_GPU(x, device)
#     graph_vec_maps[k] = prediction(model, x)

# # calculate sim, write result
# f = open("checkpoint/simgnn_result.txt", "w")
# header = "fid,fid0:sim0,fid1:sim1,fid2:sim2,fid3:sim3,fid4:sim4,fid5:sim5,fid6:sim6,fid7:sim7,fid8:sim8,fid9:sim9"
# f.write(f"{header}\n")
# querys, candidates = get_query_candidate()
# print("get sim")
# for q, candidate in tqdm(zip(querys, candidates)):
#     res = get_query_result(q, candidate, look_up=graph_vec_maps)
#     f.write(f"{res}\n")
# f.close()
