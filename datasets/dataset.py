import binascii
import csv
import json
import math
import os
import re
from collections import defaultdict

import networkx as nx

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torchvision import transforms
from tqdm import tqdm

hex2int = {hex(i)[-1]: i for i in range(16)}
asm_chars = [chr(i) for i in range(97, 97 + 26)] + [str(i) for i in range(10)] + ["[", "]", "+", ",", "\t", " ", "#"]
asm2int = {c: i for i, c in enumerate(asm_chars)}

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

    return np.array(acfg)


def preprocess_integer(asm):
    integer_regex = re.compile("(?<=\x20)-?(?!0x)[0-9]+|0[xX][0-9a-fA-F]+")
    integer_seq = integer_regex.findall(asm)

    for i, value in enumerate(integer_seq):
        if not value.startswith("0x"):
            integer_seq[i] = hex(int(value.replace("-", "")))

    for i, value in enumerate(integer_seq):
        if len(value) % 2 != 0:
            integer_seq[i] = value[:-1] + "0" + value[-1]

    integer_seq = "".join(integer_seq).replace("0x", "")
    integer_seq = np.frombuffer(binascii.unhexlify(integer_seq), np.uint8) + 1
    return integer_seq


def process_image(cfg):
    node_list = list(set(np.array(cfg).flatten()))
    edge_list = cfg
    G = nx.Graph()
    G.add_nodes_from(node_list)  # [1, 2]
    G.add_edges_from(edge_list)  # [(1, 2), (1, 3)]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize(size=(30, 30)),
        ]
    )

    img = nx.to_numpy_array(G, dtype=np.float32)
    img = np.stack((img,) * 3, axis=-1)
    return transform(img)


def process_graph(data):
    edges = data["cfg"]
    edges = np.array(edges, dtype=np.int64)

    num_of_nodes = edges.max() + 1

    features = get_acfg(data["block"], data["code"])
    if features.shape[0] < num_of_nodes:
        features = np.vstack([features, np.ones((num_of_nodes - features.shape[0], 6))])
    # features = np.ones((num_of_nodes, node_feature_dim))
    x = torch.FloatTensor(features)

    edge_index = torch.from_numpy(edges.T).type(torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data


class FuncDataset(Dataset):
    def __init__(self, args, max_len=1000, asm_max_len=4000, integer_max_len=200):
        super().__init__()
        self.args = args
        self.quicktest = args.quicktest
        self.node_feature_dim = args.node_feature_dim
        self.max_len = max_len
        self.asm_max_len = asm_max_len
        self.integer_max_len = integer_max_len

    def _preprocess(self, data):
        new_data = {}
        if self.args.wide:
            wide = hash(str(data["arch"]) + str(data["compiler"]) + str(data["opti"])) % 130
            new_data["wide"] = torch.tensor(wide, dtype=torch.long)
        if self.args.bytecode:
            new_data["code"] = self.process_code(data["byte"])
        if self.args.asm:
            new_data["asm"] = self.process_asm(data["asm"])
        if self.args.graph:
            new_data["graph"] = data["graph"]
        if self.args.integer:
            new_data["integer"] = self.process_integer(data["integer"])
        if self.args.image:
            new_data["image"] = data["image"]

        return new_data

    def process_asm(self, tokens):
        if len(tokens) < self.asm_max_len:
            tokens = np.pad(tokens, (0, self.asm_max_len - len(tokens)), "constant")
        else:
            tokens = tokens[: self.asm_max_len]
        return torch.tensor(tokens, dtype=torch.long)

    def process_code(self, bytedata):
        # convert code to bytedata
        if len(bytedata) < self.max_len:
            bytedata = np.pad(bytedata, (0, self.max_len - len(bytedata)), "constant")
        else:
            bytedata = bytedata[: self.max_len]
        return torch.tensor(bytedata, dtype=torch.long)

    def process_integer(self, integer_seq):
        if len(integer_seq) < self.integer_max_len:
            integer_seq = np.pad(integer_seq, (0, self.integer_max_len - len(integer_seq)), "constant")
        else:
            integer_seq = integer_seq[: self.integer_max_len]
        return torch.tensor(integer_seq, dtype=torch.long)

    def process_cfg(self, data):
        # convert cfg to pyg data
        edges = data
        edges = np.array(edges, dtype=np.int64)

        num_of_nodes = edges.max() + 1
        node_feature_dim = self.node_feature_dim

        features = np.ones((num_of_nodes, node_feature_dim))
        x = torch.FloatTensor(features)

        edge_index = torch.from_numpy(edges.T).type(torch.long)

        data = Data(x=x, edge_index=edge_index)
        return data


class PairDataset(FuncDataset):
    """
    func_dict, group_pair
    code + cfg
    """

    def __init__(self, args):
        super().__init__(args)
        self.node_feature_dim = args.node_feature_dim
        self.func_dict = {}

        self.get_data(args.func_file, args.group_file)

    def get_data(self, func_file, group_file):
        import pickle

        with open("data/ycc/val3_idset.pickle", "rb") as f:
            test_lyst = pickle.load(f)

        # get group data
        all_combines = []
        for _ in range(self.args.repeat):
            with open(group_file, "r") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i % 2 == 0:
                        continue
                    group_id = row[0]
                    pair = np.random.choice(row[1:], size=2, replace=True)
                    all_combines.append(pair)
                    if self.quicktest:
                        if i >= 300:
                            break

        self.all_pairs = all_combines
        selected_func_set = set(np.concatenate(self.all_pairs))
        selected_func_set = set(test_lyst) | selected_func_set

        # get func data
        print("prepare data")
        with open(func_file, "r") as f:
            for i, line in tqdm(enumerate(f)):
                pline = json.loads(line)
                # fid,arch,compiler,opti,code,block,cfg
                if len(pline["cfg"]) < 1 or str(pline["fid"]) not in selected_func_set:
                    continue
                pline["byte"] = "".join([i[1] for i in pline["code"]])
                pline["asm"] = ",".join([i[2] for i in pline["code"]])
                pline["integer"] = preprocess_integer(pline["asm"])

                pline["byte"] = np.frombuffer(binascii.unhexlify(pline["byte"]), dtype=np.uint8) + 1
                if self.args.asm:
                    pline["asm"] = np.array([ord(i) for i in pline["asm"]], dtype=np.uint8) + 1
                if self.args.graph:
                    # pline["graph"] = self.process_cfg(pline["cfg"])
                    pline["graph"] = process_graph(pline)
                if self.args.image:
                    pline["image"] = process_image(pline["cfg"])

                del pline["code"]
                del pline["block"]
                del pline["cfg"]
                self.func_dict[str(pline["fid"])] = pline
                if self.quicktest:
                    if i >= 10000:
                        break

        self.default_fid = list(self.func_dict.keys())[0]
        print("we have funcs : {}".format(len(self.func_dict)))

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        pair = self.all_pairs[idx]
        origin_data = self.func_dict.get(pair[0], self.func_dict[self.default_fid])
        pos_data = self.func_dict.get(pair[1], self.func_dict[self.default_fid])
        origin, pos = self._preprocess(origin_data), self._preprocess(pos_data)

        return origin, pos


class TestDataset(FuncDataset):
    """
    func_dict, group_pair
    code + cfg
    """

    def __init__(self, args):
        super().__init__(args)
        self.func_dict = {}
        self.all_func = []
        self.get_data(args.test_func_file)

    def get_data(self, func_file):
        import pickle

        with open("data/ycc/val3_idset.pickle", "rb") as f:
            test_lyst = pickle.load(f)
        selected_func_set = set(test_lyst)

        # get func data
        print("prepare data")
        with open(func_file, "r") as f:
            for i, line in tqdm(enumerate(f)):
                pline = json.loads(line)
                # fid,arch,compiler,opti,code,block,cfg
                if len(pline["cfg"]) < 1 or str(pline["fid"]) not in selected_func_set:
                    continue
                pline["byte"] = "".join([i[1] for i in pline["code"]])
                pline["asm"] = ",".join([i[2] for i in pline["code"]])
                pline["integer"] = preprocess_integer(pline["asm"])

                pline["byte"] = np.frombuffer(binascii.unhexlify(pline["byte"]), dtype=np.uint8) + 1
                if self.args.asm:
                    pline["asm"] = np.array([ord(i) for i in pline["asm"]], dtype=np.uint8) + 1
                if self.args.graph:
                    # pline["graph"] = self.process_cfg(pline["cfg"])
                    pline["graph"] = process_graph(pline)
                if self.args.image:
                    pline["image"] = process_image(pline["cfg"])

                del pline["code"]
                del pline["block"]
                del pline["cfg"]
                self.func_dict[str(pline["fid"])] = pline
                if self.quicktest:
                    if i >= 10000:
                        break

        self.all_func = list(self.func_dict.keys())

    def __len__(self):
        return len(self.all_func)

    def __getitem__(self, idx):
        index = self.all_func[idx]
        origin_data = self.func_dict[index]
        origin = self._preprocess(origin_data)

        return origin

    def _preprocess(self, data):
        new_data = {}
        new_data["fid"] = data["fid"]

        if self.args.wide:
            wide = hash(str(data["arch"]) + str(data["compiler"]) + str(data["opti"])) % 130
            new_data["wide"] = torch.tensor(wide, dtype=torch.long)
        if self.args.bytecode:
            new_data["code"] = self.process_code(data["byte"])
        if self.args.asm:
            new_data["asm"] = self.process_asm(data["asm"])
        if self.args.graph:
            new_data["graph"] = data["graph"]
        if self.args.integer:
            new_data["integer"] = self.process_integer(data["integer"])
        if self.args.image:
            new_data["image"] = data["image"]

        return new_data


def collate(data_list):
    wide_data = []
    code_data, asm_data, integer_data = [], [], []
    graph_data, image_data = [], []
    for data in data_list:
        for i in range(len(data)):
            if "code" in data[i]:
                code_data.append(data[i]["code"])
            if "asm" in data[i]:
                asm_data.append(data[i]["asm"])
            if "graph" in data[i]:
                graph_data.append(data[i]["graph"])
            if "wide" in data[i]:
                wide_data.append(data[i]["wide"])
            if "integer" in data[i]:
                integer_data.append(data[i]["integer"])
            if "image" in data[i]:
                image_data.append(data[i]["image"])

    if len(code_data) > 0:
        code_data = torch.vstack(code_data)
    if len(asm_data) > 0:
        asm_data = torch.vstack(asm_data)
    if len(graph_data) > 0:
        graph_data = Batch.from_data_list(graph_data)
    if len(wide_data) > 0:
        wide_data = torch.vstack(wide_data)
    if len(integer_data) > 0:
        integer_data = torch.vstack(integer_data)
    if len(image_data) > 0:
        image_data = torch.vstack(image_data)
    return {
        "code": code_data,
        "asm": asm_data,
        "graph": graph_data,
        "wide": wide_data,
        "integer": integer_data,
        "image": image_data,
    }


def test_collate(data_list):
    code_data, asm_data, integer_data = [], [], []
    graph_data, image_data = [], []
    fid, wide_data = [], []
    for data in data_list:
        if "code" in data:
            code_data.append(data["code"])
        if "asm" in data:
            asm_data.append(data["asm"])
        if "graph" in data:
            graph_data.append(data["graph"])
        if "wide" in data:
            wide_data.append(data["wide"])
        if "integer" in data:
            integer_data.append(data["integer"])
        if "image" in data:
            image_data.append(data["image"])
        fid.append(data["fid"])

    if len(code_data) > 0:
        code_data = torch.vstack(code_data)
    if len(asm_data) > 0:
        asm_data = torch.vstack(asm_data)
    if len(graph_data) > 0:
        graph_data = Batch.from_data_list(graph_data)
    if len(wide_data) > 0:
        wide_data = torch.vstack(wide_data)
    if len(integer_data) > 0:
        integer_data = torch.vstack(integer_data)
    if len(image_data) > 0:
        image_data = torch.vstack(image_data)
    return {
        "fid": fid,
        "code": code_data,
        "asm": asm_data,
        "graph": graph_data,
        "wide": wide_data,
        "integer": integer_data,
        "image": image_data,
    }


class TripletDataset(PairDataset):
    """
    func_dict, Triplet Dataset: anchor pos neg
    code + cfg
    """

    def __init__(self, args):
        super().__init__(args)
        self.all_fid = list(self.func_dict.keys())
        self.all_fid_length = len(self.all_fid)

    def __getitem__(self, idx):
        pair = self.all_pairs[idx]
        origin_data = self.func_dict.get(pair[0], self.func_dict[self.default_fid])
        pos_data = self.func_dict.get(pair[1], self.func_dict[self.default_fid])

        random_idx = np.random.randint(0, self.all_fid_length)
        neg_data = self.func_dict[self.all_fid[random_idx]]

        origin, pos = self._preprocess(origin_data), self._preprocess(pos_data)
        neg = self._preprocess(neg_data)

        return origin, pos, neg
