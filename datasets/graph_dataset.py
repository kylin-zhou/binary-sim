import binascii
import csv
import json
import math
import os
import re
import pickle
from collections import defaultdict

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Batch, Data, DataLoader, InMemoryDataset
from torch_geometric.utils import (
    add_remaining_self_loops,
    dense_to_sparse,
    remove_self_loops,
)

import sys

sys.path.append("..")
from models import BinaryModel
from .dataset import FuncDataset, preprocess_integer

from utils import ConfigParse, get_logger


device = "cuda" if torch.cuda.is_available() else "cpu"

model_file = "model.pt"

config_file = json.load(open("config.json", "r"))
config = ConfigParse(config_file)
model = BinaryModel(args=config).to(device)
model.load_state_dict(torch.load(model_file))

block_dataset = FuncDataset(
    args=config, max_len=100, asm_max_len=300, integer_max_len=20
)


def process_deep_block(code):
    byte, asm = "", ""
    for c in code:
        byte += c[0]
        asm += c[1]
    # process block asm, byte, integer
    data = {}
    data["asm"] = np.array([ord(i) for i in asm], dtype=np.uint8) + 1
    data["code"] = np.frombuffer(binascii.unhexlify(byte), dtype=np.uint8) + 1
    data["integer"] = preprocess_integer(asm)

    data["asm"] = block_dataset.process_asm(data["asm"]).to(device)
    data["code"] = block_dataset.process_code(data["code"]).to(device)
    data["integer"] = block_dataset.process_integer(data["integer"]).to(device)

    # model predict
    return data


def get_deep_acfg(block, code):
    block_code = defaultdict(list)
    bstart_index = [b[2] for b in block]

    start = 0
    try:
        for c in code:
            if c[0] < bstart_index[start]:
                block_code[start].append([c[1], c[2]])
            else:
                start += 1
                block_code[start].append([c[1], c[2]])
    except:
        pass

    acfg = []
    for idx, c in block_code.items():
        acfg.append(process_block(c))

    code_data, asm_data, integer_data = [], [], []
    for data in acfg:
        code_data.append(data["code"])
        asm_data.append(data["asm"])
        integer_data.append(data["integer"])

    code_data = torch.vstack(code_data)
    asm_data = torch.vstack(asm_data)
    integer_data = torch.vstack(integer_data)

    new_data = {
        "code": code_data,
        "asm": asm_data,
        "integer": integer_data,
    }

    with torch.no_grad():
        model.eval()
        x = model(new_data).detach().cpu().numpy()

    return x


data_regex = re.compile(
    "MOV|LDR|LDRB|LDRH|STR|STRB|STRH|MVN|PUSH|POP|XCHG|IN|OUT|XLAT|LEA|LDS|LES|LAHF|SAHF|PUSHF|POPF|\
move|movf|movt|movn|movz|mfhi|mflo|dla|la|dli|li|lw|sw|lh|lhu|sh|lb|lbu|sb|ll|sc|lui",
    re.IGNORECASE,
)

arithmetic_regex = re.compile(
    "ADD|ADC|SUB|SBC|CMP|CMN|MUL|MLA|UMULL|UMLAL|SMULL|SMLAL|\
INC|AAA|DAA|SUB|SBB|DEC|NEG|CMP|AAS|DAS|IMUL|AAM|DIV|IDIV|AAD|CBW|CWD|CWDE|CDQ|\
addi|abs|neg|div|mul|rem|mad|slt|slti|beg|bne|bgt|bge|blt|ble",
    re.IGNORECASE,
)

logic_regex = re.compile(
    "AND|ORR|EOR|BIC\
|OR|XOR|NOT|TEST|SHL|SAL|SHR|SAR|ROL|ROR|RCL|RCR\
|nor|andi|ori|sll|srl",
    re.IGNORECASE,
)

call_regex = re.compile("call", re.IGNORECASE)

jump_regex = re.compile(
    "B|BL|BX\
|JMP|RET|RETF|JA|JNBE|JAE|JNB|JB|JNAE|JBE|JNA|JG|JNLE|JGE|JNL|JL|JNGE|JLE|JNG|JE|JZ|JNE|JNZ|JC|JNC|JNO|JNP/JPO|JNS|JO|JP/JPE|JS|\
j|jr|jal",
    re.IGNORECASE,
)


def process_block(code):
    asm, opcode = "", ""
    for c in code:
        asm += c[1]
        opcode += c[1].split("\t")[0]
    data = (
        len(data_regex.findall(opcode)),
        len(arithmetic_regex.findall(opcode)),
        len(logic_regex.findall(opcode)),
        len(call_regex.findall(opcode)),
        len(jump_regex.findall(opcode)),
        len(code),
    )

    return data


def get_acfg(block, code):
    block_code = defaultdict(list)
    bstart_index = [b[2] for b in block]

    start = 0
    try:
        for c in code:
            if c[0] < bstart_index[start]:
                block_code[start].append([c[1], c[2]])
            else:
                start += 1
                block_code[start].append([c[1], c[2]])
    except:
        pass

    acfg = []
    for idx, c in block_code.items():
        acfg.append(process_block(c))

    return acfg


def process_graph(cfg, block, code):
    edges = cfg
    edges = np.array(edges, dtype=np.int64)

    num_of_nodes = edges.max() + 1
    node_feature_dim = 64

    features = get_acfg(block, code)
    if features.shape[0] < num_of_nodes:
        features = np.vstack(
            [features, np.ones((num_of_nodes - features.shape[0], node_feature_dim))]
        )
    x = torch.FloatTensor(features)

    edge_index = torch.from_numpy(edges.T).type(torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data


class BinaryFuncDataset(InMemoryDataset):
    def __init__(
        self,
        func_file,
        train_group_file,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.func_file = func_file
        self.train_group_file = train_group_file
        self.number_features = 64
        self.func2graph = dict()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.func2graph, self.number_features = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def get_group(self):
        file = self.train_group_file
        fid2group = {}
        self.all_pairs = []
        with open(file, "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                group_id = row[0]
                group_fids = row[1:]
                for fid in group_fids:
                    fid2group[int(fid)] = int(group_id)
                if i >= 1000:
                    break

        return fid2group

    def process(self):
        fid2group = self.get_group()
        print("all fids: ", len(fid2group))
        with open(self.func_file, "r") as f:
            for i, line in tqdm(enumerate(f)):
                pline = json.loads(line)
                fid, cfg = pline["fid"], pline["cfg"]
                if len(cfg) < 1 or fid not in fid2group:
                    continue
                codes, blocks = pline["code"], pline["block"]
                group_id = fid2group[fid]
                data = process_graph(cfg=cfg, block=blocks, code=codes)

                if group_id in self.func2graph:
                    self.func2graph[group_id].append(data)
                else:
                    self.func2graph[group_id] = [data]

        torch.save((self.func2graph, self.number_features), self.processed_paths[0])


class GraphEmbeddingDataset(object):
    def __init__(self, args):
        self.args = args
        self.training_funcs = dict()
        self.validation_funcs = dict()
        self.testing_funcs = dict()
        self.number_features = None
        self.id2name = None
        self.func2graph = None
        self.process_dataset()

    def process_dataset(self):
        print("\nPreparing datasets.\n")
        self.dataset = BinaryFuncDataset(
            func_file="data/public/train.func.json",
            train_group_file="data/public/train.group.csv",
            root="data/",
        )
        self.number_features = self.dataset.number_features
        self.func2graph = self.dataset.func2graph
        self.id2name = dict()

        cnt = 0
        for k, v in self.func2graph.items():
            self.id2name[cnt] = k
            cnt += 1

        self.train_num = int(len(self.func2graph) * 0.8)
        self.val_num = int(len(self.func2graph) * 0.1)
        self.test_num = int(len(self.func2graph)) - (self.train_num + self.val_num)

        random_idx = np.random.permutation(len(self.func2graph))
        self.train_idx = random_idx[0 : self.train_num]
        self.val_idx = random_idx[self.train_num : self.train_num + self.val_num]
        self.test_idx = random_idx[self.train_num + self.val_num :]

        self.training_funcs = self.split_dataset(self.training_funcs, self.train_idx)
        self.validation_funcs = self.split_dataset(self.validation_funcs, self.val_idx)
        self.testing_funcs = self.split_dataset(self.testing_funcs, self.test_idx)

    def split_dataset(self, funcdict, idx):
        for i in idx:
            funcname = self.id2name[i]
            funcdict[funcname] = self.func2graph[funcname]
        return funcdict

    def collate(self, data_list):
        code_data, asm_data, graph_data, wide_data = [], [], [], []
        for data in data_list:
            for i in range(len(data)):
                graph_data.append(data[i])

        return Batch.from_data_list(graph_data)

    def create_batches(self, funcs, collate, shuffle_batch=True):
        data = FuncDataset(funcs)
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=self.args.batch_size,
            shuffle=shuffle_batch,
            collate_fn=collate,
            num_workers=8,
            pin_memory=True,
        )

        return loader


class FuncDataset(Dataset):
    def __init__(self, funcdict):
        super(FuncDataset, self).__init__()
        self.funcdict = funcdict
        self.id2key = dict()  # idx to groupid
        cnt = 0
        for k, v in self.funcdict.items():
            self.id2key[cnt] = k
            cnt += 1

    def __len__(self):
        return len(self.funcdict)

    def get_graph_pair(self, graphset):
        count = 0
        while True:
            pos_idx = np.random.choice(range(len(graphset)), size=2, replace=True)
            origin_graph = graphset[pos_idx[0]]
            pos_graph = graphset[pos_idx[1]]
            # if cfg diff > 0.3*min, repeat sample
            if np.absolute(len(origin_graph) - len(pos_graph)) < int(
                0.3 * min(len(origin_graph), len(pos_graph))
            ):
                break
            # force quit
            count += 1
            if count > 10:
                break
        return origin_graph, pos_graph

    def __getitem__(self, idx):
        # idx to groupid, groupid to funcset
        graphset = self.funcdict[self.id2key[idx]]
        origin_graph, pos_graph = self.get_graph_pair(graphset)
        return origin_graph, pos_graph
