import binascii
import csv
import json
from collections.abc import Iterable

import networkx as nx
import numpy as np
from tqdm import tqdm

# node/graph vector
def _normalize(raw_obj):
    counts = np.array(raw_obj, dtype=np.float32)
    sum = counts.sum()
    normalized = counts / sum
    return normalized


def hex_index():
    hex16 = [hex(i)[-1] for i in range(16)]
    hex256 = [i + j for i in hex16 for j in hex16]
    maps = {k: v for k, v in zip(hex256, range(len(hex256)))}
    return maps


def lookup():
    np.random.seed(0)
    tables = np.random.randn(256, 32)
    return tables


def blocks_embedding(block, asm):
    all_block_vec = []
    for b in range(len(block)):
        block_code = ""
        for i in range(block[b][1], block[b][2]):
            for j in range(len(asm)):
                if asm[j][0] == i:
                    block_code += asm[j][1]

        hexdata = [block_code[i : i + 2] for i in range(0, len(block_code), 2)]
        block_vec = np.sum([lookup_table[hex2index[i]] for i in hexdata], axis=0)
        if isinstance(block_vec, Iterable) and len(block_vec) > 0:
            all_block_vec.append(block_vec)
    return np.sum(np.array(all_block_vec), axis=0)


def get_graph(cfg):
    node_list = list(set(np.array(cfg).flatten()))
    edge_list = cfg
    G = nx.Graph()
    G.add_nodes_from(node_list)  # [1, 2]
    G.add_edges_from(edge_list)  # [(1, 2), (1, 3)]
    return G


# ndoes, edges, components, degree(sum,mean,max)
def graph_feature(cfg):
    g = get_graph(cfg)
    n_nodes, n_edges, n_components = (
        g.number_of_nodes(),
        g.number_of_edges(),
        nx.number_connected_components(g),
    )
    degrees = [d for n, d in g.degree()]
    if len(degrees) > 0:
        maxd, sumd, meand = max(degrees), sum(degrees), sum(degrees) / len(degrees)
    else:
        maxd, sumd, meand = 0, 0, 0
    vec = [n_nodes, n_edges, n_components, maxd, sumd, meand]
    return np.array(vec)


def grpah_embedding(g):
    asm, blocks, cfg = g
    f1 = blocks_embedding(blocks, asm)
    f2 = graph_feature(cfg)
    return np.hstack([f1, f2])


# cosine sim
def pairwise_dot_product_similarity(x, y):
    # x.shape [1,dim], y.shape [n,dim]
    dot = x * y
    factor = np.linalg.norm(x) * np.linalg.norm(y, axis=1)
    return np.sum(x * y, axis=1) / factor


func_file = "datadata/public/test.func.json"

with open(func_file, "r") as f:
    lines = f.readlines()

test_func_maps = {}
for line in tqdm(lines):
    pline = json.loads(line)
    test_func_maps[pline["fid"]] = pline


file = "datadata/public/test.question.csv"

querys = []
candidates = []

with open(file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        querys.append(int(row[0]))
        candidates.append([int(i) for i in row[1:]])


def get_query_result(query, candidate, look_up):
    vec = []
    for a in candidate:
        vec.append(look_up.get(a, np.zeros(38)))
    answer_vecs = np.stack(vec, axis=0)
    query_vec = look_up[query]
    sim = pairwise_dot_product_similarity(query_vec, answer_vecs)

    index = sorted(
        [[i, v] for i, v in enumerate(sim)], key=lambda x: x[1], reverse=True
    )[:10]
    res = str(q) + "," + ",".join([f"{candidate[i[0]]}:{i[1]}" for i in index])
    return res


def process_cfg(query_data):
    x = query_data["code"], query_data["block"], query_data["cfg"]
    return x


hex2index = hex_index()
lookup_table = lookup()

# get graph embedding
print("get func")
func_maps = test_func_maps
print("get graph embdding")
graph_vec_maps = {}
for k, v in tqdm(func_maps.items()):
    x = process_cfg(v)
    graph_vec_maps[k] = grpah_embedding(x)

# calculate sim, write result
f = open("checkpoint/result.txt", "w")
header = "fid,fid0:sim0,fid1:sim1,fid2:sim2,fid3:sim3,fid4:sim4,fid5:sim5,fid6:sim6,fid7:sim7,fid8:sim8,fid9:sim9"
f.write(f"{header}\n")
print("get sim")
for q, candidate in tqdm(zip(querys, candidates)):
    res = get_query_result(q, candidate, look_up=graph_vec_maps)
    f.write(f"{res}\n")
f.close()
