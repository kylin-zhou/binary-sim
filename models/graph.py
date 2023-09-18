# -*- coding: UTF-8 -*-
# Date: 2022/09/19

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    GINConv,
    MessagePassing,
    global_add_pool,
    global_mean_pool,
)


class ReadoutModule(torch.nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(ReadoutModule, self).__init__()
        self.args = args

        self.weight = torch.nn.Parameter(torch.Tensor(self.args.nhid, self.args.nhid))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x, batch):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: Result of the GNN.
        :param batch: Batch vector, which assigns each node to a specific example
        :param size: Size
        :return representation: A graph level representation matrix.
        """
        mean_pool = global_mean_pool(x, batch)
        transformed_global = torch.tanh(torch.mm(mean_pool, self.weight))
        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x

        return global_add_pool(weighted, batch)


class GCN(nn.Module):
    """
    GraphEmbedding; base class
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.node_feature_dim = args.node_feature_dim
        self.nhid = args.nhid

        self.convolution_1 = GCNConv(self.node_feature_dim, self.nhid)
        self.convolution_2 = GCNConv(self.nhid, self.nhid)
        self.linear = nn.Linear(self.nhid, self.nhid)
        self.readout = ReadoutModule(self.args)

    def convolutional_pass(self, features, edge_index):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = F.relu(features)
        features = self.linear(features)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)

        features = self.convolution_2(features, edge_index)

        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return embedding: .
        """
        edge_index, features = data.edge_index, data.x
        batch = data.batch

        abstract_features = self.convolutional_pass(features, edge_index)
        pooled_features = self.readout(abstract_features, batch)

        return pooled_features


class GIN(torch.nn.Module):
    """GIN
    https://mlabonne.github.io/blog/gin/
    """

    def __init__(self, args):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(args.node_feature_dim, args.nhid),
                nn.BatchNorm1d(args.nhid),
                nn.ReLU(),
                nn.Linear(args.nhid, args.nhid),
                nn.ReLU(),
            )
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(args.nhid, args.nhid),
                nn.BatchNorm1d(args.nhid),
                nn.ReLU(),
                nn.Linear(args.nhid, args.nhid),
                nn.ReLU(),
            )
        )
        self.readout = ReadoutModule(args)
        self.lin1 = nn.Linear(args.nhid * 2, args.nhid)

    def forward(self, data):
        edge_index, features = data.edge_index, data.x
        batch = data.batch

        # Node embeddings
        h1 = self.conv1(features, edge_index)
        h2 = self.conv2(h1, edge_index)

        # Graph-level readout
        h1 = self.readout(h1, batch)
        h2 = self.readout(h2, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2), dim=1)

        # Classifier
        h = self.lin1(h)

        return h
