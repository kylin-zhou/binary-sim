import abc
import collections
import contextlib
import copy
import json
import random
import csv
import binascii
from collections.abc import Iterable
from tqdm import tqdm
from functools import reduce
from itertools import combinations

import numpy as np

"""A general Interface"""

# node/graph vector


def hex_index():
    hex16 = [hex(i)[-1] for i in range(16)]
    hex256 = [i + j for i in hex16 for j in hex16]
    maps = {k: v for k, v in zip(hex256, range(len(hex256)))}
    return maps


def lookup():
    np.random.seed(0)
    tables = np.random.randn(256, 32)
    return tables


def node_block_embedding(block, code):
    all_block_vec = []
    for b in range(len(block)):
        block_code = ""
        for i in range(block[b][1], block[b][2]):
            for j in range(len(code)):
                if code[j][0] == i:
                    block_code += code[j][1]

        hexdata = [block_code[i : i + 2] for i in range(0, len(block_code), 2)]
        block_vec = np.sum([lookup_table[hex2index[i]] for i in hexdata], axis=0)
        if isinstance(block_vec, Iterable) and len(block_vec) > 0:
            all_block_vec.append(block_vec)
        else:
            all_block_vec.append(np.zeros(32))
    return np.stack(all_block_vec, axis=0)


hex2index = hex_index()
lookup_table = lookup()


class GraphEditDistanceDataset:
    """Graph edit distance dataset."""

    def __init__(
        self,
        train_func_file="data/public/train.func.json",
        train_group_file="data/public/train.group.csv",
        node_feature_dim=16,
        edge_feature_dim=8,
    ):
        """Constructor.
        Args:
        """
        self.train_func_file = train_func_file
        self.train_group_file = train_group_file
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.func_maps = self.get_func_data()
        self.fids = list(self.func_maps.keys())
        combine_pairs = self.get_combine_pair()
        print("number of all pairs: ", len(combine_pairs))
        random.shuffle(combine_pairs)
        valid_index = int(len(combine_pairs) * 0.9)
        self.train_combine_pairs = combine_pairs[:valid_index]
        self.valid_combine_pairs = combine_pairs[valid_index:]
        print("train number of pairs: ", len(self.train_combine_pairs))
        self.negative = 5

    def get_func_data(self):
        print("process function data")
        func_file = self.train_func_file
        with open(func_file, "r") as f:
            lines = f.readlines()
        print("start dump func map")
        func_maps = {}
        for line in tqdm(lines):
            pline = json.loads(line)
            if len(pline["cfg"]) > 1:
                func_maps[str(pline["fid"])] = pline["cfg"]
        return func_maps

    def get_combine_pair(self):
        print("genrate pair data")
        file = self.train_group_file
        all_combines = []
        with open(file, "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                group_id = row[0]
                combines = list(combinations(row[1:], 2))
                all_combines.extend(combines)
                if i == 3:
                    print("break")
                    break
        return all_combines

    def get_pair(self):
        """Generate one pair of graphs."""
        while True:
            if self.valid:
                combines = self.valid_combine_pairs
                random.shuffle(combines)
            else:
                combines = self.train_combine_pairs
                random.shuffle(combines)
            for pair in combines:
                cfg1 = self.func_maps.get(pair[0], "-1")
                for i in range(self.negative):
                    if self.positive:
                        cfg2 = self.func_maps.get(pair[1], "-1")
                    else:
                        random_id = random.sample(self.fids, 1)[0]
                        cfg2 = self.func_maps.get(random_id, "-1")
                    # if max(len(cfg1), len(cfg2)) > 2 * min(len(cfg1), len(cfg2)):
                    #     continue
                    yield cfg1, cfg2

    def pairs(self, batch_size, valid=False):
        """Yields batches of pair data."""
        self.valid = valid
        pair_gen = self.get_pair()
        while True:
            batch_graphs = []
            batch_labels = []
            self.positive = False
            for i in range(batch_size):
                g1, g2 = next(pair_gen)
                batch_graphs.append((g1, g2))
                batch_labels.append(1 if self.positive else -1)
                self.positive = (
                    not self.positive if i % self.negative == 0 else self.positive
                )

            packed_graphs = self._pack_batch(batch_graphs)
            labels = np.array(batch_labels, dtype=np.int32)
            yield packed_graphs, labels

    def get_node_feature(self, block, code):
        node_features = node_block_embedding(block, code)
        return node_features

    def _pack_batch(self, graphs):
        """Pack a batch of graphs into a single `GraphData` instance.
        Args:
          graphs: a list of generated networkx graphs.
        Returns:
          graph_data: a `GraphData` instance, with node and edge indices properly
            shifted.
        """
        Graphs = []
        for graph in graphs:
            for inergraph in graph:
                Graphs.append(inergraph)
        graphs = Graphs
        from_idx = []
        to_idx = []
        graph_idx = []
        node_feature_dim = self.node_feature_dim
        edge_feature_dim = self.edge_feature_dim

        n_total_nodes = 0
        n_total_edges = 0
        for i, g in enumerate(graphs):
            edges = np.array(g, dtype=np.int32)
            try:
                n_nodes = edges.max() + 1
                n_edges = len(edges)

                # shift the node indices for the edges
                from_idx.append(edges[:, 0] + n_total_nodes)
                to_idx.append(edges[:, 1] + n_total_nodes)
                graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

                n_total_nodes += n_nodes
                n_total_edges += n_edges
            except:
                print("warning: ~~~")
                print(g)
                continue

        GraphData = collections.namedtuple(
            "GraphData",
            [
                "from_idx",
                "to_idx",
                "node_features",
                "edge_features",
                "graph_idx",
                "n_graphs",
            ],
        )

        graph_data = GraphData(
            from_idx=np.concatenate(from_idx, axis=0),
            to_idx=np.concatenate(to_idx, axis=0),
            # this task only cares about the structures, the graphs have no features.
            # setting higher dimension of ones to confirm code functioning
            # with high dimensional features.
            node_features=np.ones((n_total_nodes, node_feature_dim), dtype=np.float32),
            edge_features=np.ones((n_total_edges, edge_feature_dim), dtype=np.float32),
            graph_idx=np.concatenate(graph_idx, axis=0),
            n_graphs=len(graphs),
        )

        return graph_data


class TestGraphEditDistanceDataset:
    """A fixed dataset of pairs or triplets for the graph edit distance task.
    This dataset can be used for evaluation.
    """

    def __init__(
        self,
        func_file="data/public/test.func.json",
        question_file="data/public/test.question.csv",
        node_feature_dim=16,
        edge_feature_dim=8,
    ):
        """Constructor.
        Args:
        """
        self.func_file = func_file
        self.question_file = question_file
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.func_maps = self.get_group_func_data()
        self.all_graphs = list(self.func_maps.keys())

    def get_group_func_data(self):
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
        return test_func_maps

    def get_func_data(self):
        print("process function data")
        test_func_maps = {}
        with open(self.func_file, "r") as f:
            lines = f.readlines()
        test_func_maps = {}
        for line in tqdm(lines):
            pline = json.loads(line)
            cfg = pline["cfg"]
            if len(cfg) > 1:
                test_func_maps[str(pline["fid"])] = cfg
        return test_func_maps

    def get_cfg_graph(self, fid):
        return self.func_maps[fid]

    def single(self, batch_size):
        index = 0
        while index < len(self.all_graphs):
            batch_graphs = []
            batch_ids = []
            for _ in range(batch_size):
                if index < len(self.all_graphs):
                    g = self.get_cfg_graph(self.all_graphs[index])

                    batch_ids.append(self.all_graphs[index])
                    batch_graphs.append(g)
                    index += 1

            packed_graphs = self._pack_batch(batch_graphs)
            yield batch_ids, packed_graphs

    def _pack_batch(self, graphs):
        """Pack a batch of graphs into a single `GraphData` instance.
        Args:
          graphs: a list of generated networkx graphs.
        Returns:
          graph_data: a `GraphData` instance, with node and edge indices properly
            shifted.
        """
        Graphs = []
        for graph in graphs:
            Graphs.append(graph)
        graphs = Graphs
        from_idx = []
        to_idx = []
        graph_idx = []
        node_feature_dim = self.node_feature_dim
        edge_feature_dim = self.edge_feature_dim

        n_total_nodes = 0
        n_total_edges = 0
        for i, g in enumerate(graphs):
            edges = np.array(g, dtype=np.int32)
            try:
                n_nodes = edges.max() + 1
                n_edges = len(edges)

                # shift the node indices for the edges
                from_idx.append(edges[:, 0] + n_total_nodes)
                to_idx.append(edges[:, 1] + n_total_nodes)
                graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

                n_total_nodes += n_nodes
                n_total_edges += n_edges
            except:
                print("warning: ~~~")
                print(g)
                continue

        GraphData = collections.namedtuple(
            "GraphData",
            [
                "from_idx",
                "to_idx",
                "node_features",
                "edge_features",
                "graph_idx",
                "n_graphs",
            ],
        )

        graph_data = GraphData(
            from_idx=np.concatenate(from_idx, axis=0),
            to_idx=np.concatenate(to_idx, axis=0),
            # this task only cares about the structures, the graphs have no features.
            # setting higher dimension of ones to confirm code functioning
            # with high dimensional features.
            node_features=np.random.randn(n_total_nodes, node_feature_dim).astype(
                np.float32
            ),
            edge_features=np.random.randn(n_total_edges, edge_feature_dim).astype(
                np.float32
            ),
            graph_idx=np.concatenate(graph_idx, axis=0),
            n_graphs=len(graphs),
        )

        return graph_data
