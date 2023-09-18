import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HypergraphConv, RGCNConv

from layers import (
    MLP,
    AttentionModule,
    CrossGraphConvolution,
    HyperedgeConv,
    HyperedgePool,
    MLPModule,
    ReadoutModule,
)
from utils import hypergraph_construction


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.nhid = args.nhid
        self.k = args.k
        self.mode = args.mode

        self.num_features = args.num_features

        self.conv0 = GCNConv(self.num_features, self.nhid)
        self.conv1 = HypergraphConv(self.nhid, self.nhid)
        self.cross_conv1 = CrossGraphConvolution(self.nhid, self.nhid)
        self.pool1 = HyperedgePool(self.nhid, self.args.ratio1)

        self.conv2 = HyperedgeConv(self.nhid, self.nhid)
        self.cross_conv2 = CrossGraphConvolution(self.nhid, self.nhid)
        self.pool2 = HyperedgePool(self.nhid, self.args.ratio2)

        self.conv3 = HyperedgeConv(self.nhid, self.nhid)
        self.cross_conv3 = CrossGraphConvolution(self.nhid, self.nhid)
        self.pool3 = HyperedgePool(self.nhid, self.args.ratio3)

        self.readout0 = ReadoutModule(self.args)
        self.readout1 = ReadoutModule(self.args)
        self.readout2 = ReadoutModule(self.args)
        self.readout3 = ReadoutModule(self.args)

        self.mlp = MLPModule(self.args)

    def forward(self, data):
        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index

        edge_attr_1 = data["g1"].edge_attr
        edge_attr_2 = data["g2"].edge_attr

        features_1 = data["g1"].x
        features_2 = data["g2"].x

        batch_1 = data["g1"].batch
        batch_2 = data["g2"].batch

        # Layer 0
        # Graph Convolution Operation
        f1_conv0 = F.leaky_relu(
            self.conv0(features_1, edge_index_1, edge_attr_1), negative_slope=0.2
        )
        f2_conv0 = F.leaky_relu(
            self.conv0(features_2, edge_index_2, edge_attr_2), negative_slope=0.2
        )

        att_f1_conv0 = self.readout0(f1_conv0, batch_1)
        att_f2_conv0 = self.readout0(f2_conv0, batch_2)
        # print(f1_conv0.shape, att_f1_conv0.shape)
        score0 = torch.cat([att_f1_conv0, att_f2_conv0], dim=1)

        edge_index_1, edge_attr_1 = hypergraph_construction(
            edge_index_1,
            edge_attr_1,
            num_nodes=features_1.size(0),
            k=self.k,
            mode=self.mode,
        )
        edge_index_2, edge_attr_2 = hypergraph_construction(
            edge_index_2,
            edge_attr_2,
            num_nodes=features_2.size(0),
            k=self.k,
            mode=self.mode,
        )

        # Layer 1
        # Hypergraph Convolution Operation
        f1_conv1 = F.leaky_relu(
            self.conv1(f1_conv0, edge_index_1, edge_attr_1), negative_slope=0.2
        )
        f2_conv1 = F.leaky_relu(
            self.conv1(f2_conv0, edge_index_2, edge_attr_2), negative_slope=0.2
        )

        # Hyperedge Pooling
        (
            edge1_conv1,
            edge1_index_pool1,
            edge1_attr_pool1,
            edge1_batch_pool1,
        ) = self.pool1(f1_conv1, batch_1, edge_index_1, edge_attr_1)
        (
            edge2_conv1,
            edge2_index_pool1,
            edge2_attr_pool1,
            edge2_batch_pool1,
        ) = self.pool1(f2_conv1, batch_2, edge_index_2, edge_attr_2)

        # Cross Graph Convolution
        hyperedge1_cross_conv1, hyperedge2_cross_conv1 = self.cross_conv1(
            edge1_conv1, edge1_batch_pool1, edge2_conv1, edge2_batch_pool1
        )

        # Readout Module
        att_f1_conv1 = self.readout1(hyperedge1_cross_conv1, edge1_batch_pool1)
        att_f2_conv1 = self.readout1(hyperedge2_cross_conv1, edge2_batch_pool1)
        score1 = torch.cat([att_f1_conv1, att_f2_conv1], dim=1)

        # Layer 2
        # Hypergraph Convolution Operation
        f1_conv2 = F.leaky_relu(
            self.conv2(hyperedge1_cross_conv1, edge1_index_pool1, edge1_attr_pool1),
            negative_slope=0.2,
        )
        f2_conv2 = F.leaky_relu(
            self.conv2(hyperedge2_cross_conv1, edge2_index_pool1, edge2_attr_pool1),
            negative_slope=0.2,
        )

        # Hyperedge Pooling
        (
            edge1_conv2,
            edge1_index_pool2,
            edge1_attr_pool2,
            edge1_batch_pool2,
        ) = self.pool2(f1_conv2, edge1_batch_pool1, edge1_index_pool1, edge1_attr_pool1)
        (
            edge2_conv2,
            edge2_index_pool2,
            edge2_attr_pool2,
            edge2_batch_pool2,
        ) = self.pool2(f2_conv2, edge2_batch_pool1, edge2_index_pool1, edge2_attr_pool1)

        # Cross Graph Convolution
        hyperedge1_cross_conv2, hyperedge2_cross_conv2 = self.cross_conv2(
            edge1_conv2, edge1_batch_pool2, edge2_conv2, edge2_batch_pool2
        )

        # Readout Module
        att_f1_conv2 = self.readout2(hyperedge1_cross_conv2, edge1_batch_pool2)
        att_f2_conv2 = self.readout2(hyperedge2_cross_conv2, edge2_batch_pool2)
        score2 = torch.cat([att_f1_conv2, att_f2_conv2], dim=1)

        # Layer 3
        # Hypergraph Convolution Operation
        f1_conv3 = F.leaky_relu(
            self.conv3(hyperedge1_cross_conv2, edge1_index_pool2, edge1_attr_pool2),
            negative_slope=0.2,
        )
        f2_conv3 = F.leaky_relu(
            self.conv3(hyperedge2_cross_conv2, edge2_index_pool2, edge2_attr_pool2),
            negative_slope=0.2,
        )

        # Hyperedge Pooling
        (
            edge1_conv3,
            edge1_index_pool3,
            edge1_attr_pool3,
            edge1_batch_pool3,
        ) = self.pool3(f1_conv3, edge1_batch_pool2, edge1_index_pool2, edge1_attr_pool2)
        (
            edge2_conv3,
            edge2_index_pool3,
            edge2_attr_pool3,
            edge2_batch_pool3,
        ) = self.pool3(f2_conv3, edge2_batch_pool2, edge2_index_pool2, edge2_attr_pool2)

        # Cross Graph Convolution
        hyperedge1_cross_conv3, hyperedge2_cross_conv3 = self.cross_conv3(
            edge1_conv3, edge1_batch_pool3, edge2_conv3, edge2_batch_pool3
        )

        # Readout Module
        att_f1_conv3 = self.readout3(hyperedge1_cross_conv3, edge1_batch_pool3)
        att_f2_conv3 = self.readout3(hyperedge2_cross_conv3, edge2_batch_pool3)
        score3 = torch.cat([att_f1_conv3, att_f2_conv3], dim=1)

        scores = torch.cat([score0, score1, score2, score3], dim=1)
        sim = self.mlp(scores)

        return scores, sim


class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.nhid = args.nhid
        self.k = args.k
        self.mode = args.mode

        self.num_features = args.num_features

        self.conv0 = RGCNConv(self.num_features, self.nhid)
        self.conv1 = HypergraphConv(self.nhid, self.nhid)
        self.pool1 = HyperedgePool(self.nhid, self.args.ratio1)

        self.readout0 = ReadoutModule(self.args)
        self.readout1 = ReadoutModule(self.args)

        self.mlp = MLP(self.args)

    def forward(self, data):
        edge_index_1 = data["g1"].edge_index
        edge_index_2 = data["g2"].edge_index

        edge_attr_1 = data["g1"].edge_attr
        edge_attr_2 = data["g2"].edge_attr

        features_1 = data["g1"].x
        features_2 = data["g2"].x

        batch_1 = data["g1"].batch
        batch_2 = data["g2"].batch

        # Layer 0
        # Graph Convolution Operation
        f1_conv0 = F.leaky_relu(
            self.conv0(features_1, edge_index_1, edge_attr_1), negative_slope=0.2
        )
        f2_conv0 = F.leaky_relu(
            self.conv0(features_2, edge_index_2, edge_attr_2), negative_slope=0.2
        )

        att_f1_conv0 = self.readout0(f1_conv0, batch_1)
        att_f2_conv0 = self.readout0(f2_conv0, batch_2)
        score0 = torch.cat([att_f1_conv0, att_f2_conv0], dim=1)

        edge_index_1, edge_attr_1 = hypergraph_construction(
            edge_index_1,
            edge_attr_1,
            num_nodes=features_1.size(0),
            k=self.k,
            mode=self.mode,
        )
        edge_index_2, edge_attr_2 = hypergraph_construction(
            edge_index_2,
            edge_attr_2,
            num_nodes=features_2.size(0),
            k=self.k,
            mode=self.mode,
        )

        # Layer 1
        # Hypergraph Convolution Operation
        f1_conv1 = F.leaky_relu(
            self.conv1(f1_conv0, edge_index_1, edge_attr_1), negative_slope=0.2
        )
        f2_conv1 = F.leaky_relu(
            self.conv1(f2_conv0, edge_index_2, edge_attr_2), negative_slope=0.2
        )

        # Readout Module
        att_f1_conv1 = self.readout1(f1_conv1, batch_1)
        att_f2_conv1 = self.readout1(f2_conv1, batch_2)
        score1 = torch.cat([att_f1_conv1, att_f2_conv1], dim=1)

        scores = torch.cat([score0, score1], dim=1)
        sim = self.mlp(scores)

        return scores, sim


class GraphEmbedding(nn.Module):
    """
    GraphEmbedding; base class
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.nhid = args.nhid
        self.k = args.k
        self.mode = args.mode
        self.num_features = args.num_features

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = RGCNConv(self.num_features, self.nhid)
        self.convolution_2 = RGCNConv(self.nhid, self.nhid)
        self.attention = AttentionModule(self.args)

    def convolutional_pass(self, features, edge_index):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(
            features, p=self.args.dropout, training=self.training
        )

        features = self.convolution_2(features, edge_index)

        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return embedding: .
        """
        edge_index, features = data.edge_index, data.x

        abstract_features = self.convolutional_pass(edge_index, features)
        pooled_features = self.attention(abstract_features)

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
        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(args.nhid, args.nhid),
                nn.BatchNorm1d(args.nhid),
                nn.ReLU(),
                nn.Linear(args.nhid, args.nhid),
                nn.ReLU(),
            )
        )
        self.readout = ReadoutModule(args)
        self.lin1 = nn.Linear(args.nhid * 3, args.nhid)

    def forward(self, data):
        edge_index, features = data.edge_index, data.x
        batch = data.batch

        # Node embeddings
        h1 = self.conv1(features, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = self.readout(h1, batch)
        h2 = self.readout(h2, batch)
        h3 = self.readout(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)

        return h

