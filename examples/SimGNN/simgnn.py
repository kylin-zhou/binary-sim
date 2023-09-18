"""SimGNN class and runner."""

import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from layers import AttentionModule, TenorNetworkModule
from utils import (
    process_pair,
    calculate_loss,
    calculate_normalized_ged,
    assign_GPU,
    pairwise_loss,
)


class GraphEmbedding(nn.Module):
    """
    GraphEmbedding; base class of SimGNN
    """

    def __init__(self, args, node_feature_dim):
        """
        :param args: Arguments object.
        :param node_feature_dim: Number of node labels. node feature dim
        """
        super().__init__()
        self.args = args
        self.node_feature_dim = node_feature_dim
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.node_feature_dim, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(
            self.feature_count, self.args.bottle_neck_neurons
        )
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def convolutional_pass(self, edge_index, features):
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
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(
            features, p=self.args.dropout, training=self.training
        )

        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return embedding: .
        """
        # edge_index, features = data["edge_index"], data["features"]
        edge_index, features = data[0], data[1]

        abstract_features = self.convolutional_pass(edge_index, features)
        pooled_features = self.attention(abstract_features)

        return abstract_features, pooled_features


class SimGNN(GraphEmbedding):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args, node_feature_dim, loss="pair"):
        super().__init__(args, node_feature_dim)
        self.loss = loss
        self.graph_embedding = GraphEmbedding(args, node_feature_dim)

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist / torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        abstract_features_1, pooled_features_1 = self.graph_embedding(
            [edge_index_1, features_1]
        )

        abstract_features_2, pooled_features_2 = self.graph_embedding(
            [edge_index_2, features_2]
        )

        if self.args.histogram == True:
            hist = self.calculate_histogram(
                abstract_features_1, torch.t(abstract_features_2)
            )

        if self.loss == "mlp":
            scores = self.tensor_network(pooled_features_1, pooled_features_2)
            scores = torch.t(scores)

            if self.args.histogram == True:
                scores = torch.cat((scores, hist), dim=1).view(1, -1)

            scores = torch.nn.functional.relu(self.fully_connected_first(scores))
            score = torch.sigmoid(self.scoring_layer(scores))
            return score
        else:
            return pooled_features_1, pooled_features_2


class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initial_label_enumeration()
        self.setup_model()

    def setup_model(self):
        """
        Creating a SimGNN.
        """
        self.model = SimGNN(self.args, self.node_feature_dim, loss=self.loss).to(
            self.device
        )

    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers.
        """
        print("\nEnumerating unique labels.\n")
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        graph_pairs = self.training_graphs + self.testing_graphs
        self.node_feature_dim = 32
        self.loss = "pair"

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph : graph + self.args.batch_size])
        return batches

    def transfer_to_torch(self, data):
        """
        Transferring the data to torch and creating a hash table.
        Including the indices, features and target.
        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()
        edges_1 = data["graph_1"]
        edges_2 = data["graph_2"]
        edges_1 = np.array(edges_1, dtype=np.int64)
        edges_2 = np.array(edges_2, dtype=np.int64)

        graph1_nodes = edges_1.max() + 1
        graph2_nodes = edges_2.max() + 1

        edges_1 = torch.from_numpy(edges_1.T).type(torch.long)
        edges_2 = torch.from_numpy(edges_2.T).type(torch.long)

        features_1 = np.ones((graph1_nodes, self.node_feature_dim))
        features_2 = np.ones((graph2_nodes, self.node_feature_dim))

        features_1 = torch.FloatTensor(np.array(features_1))
        features_2 = torch.FloatTensor(np.array(features_2))

        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2

        new_data["features_1"] = features_1
        new_data["features_2"] = features_2

        new_data["target"] = (
            torch.from_numpy(np.array(data["ged"]).reshape(1, 1)).view(-1).float()
        )
        # norm_ged = data["ged"] / (0.5 * (len(data["labels_1"]) + len(data["labels_2"])))

        # new_data["target"] = (
        #     torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float()
        # )
        return new_data

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = 0
        for graph_pair in batch:
            pair_gen = process_pair(graph_pair)
            while True:
                try:
                    data = next(pair_gen)
                    data = self.transfer_to_torch(data)
                    data = assign_GPU(data, self.device)
                    prediction = self.model(data)
                    # print(data["target"], prediction)
                    if self.loss == "pair":
                        pair_loss = pairwise_loss(
                            prediction[0].view(1, -1),
                            prediction[1].view(1, -1),
                            data["target"],
                            loss_type="margin",
                            margin=1.0,
                        )
                        losses = losses + pair_loss
                    else:
                        mlp_loss = torch.nn.functional.mse_loss(
                            data["target"], prediction
                        )
                        losses = losses + mlp_loss
                except:
                    break
        losses.backward(torch.ones_like(losses), retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        for epoch in epochs:
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0
            self.model.train()
            for index, batch in tqdm(
                enumerate(batches), total=len(batches), desc="Batches"
            ):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum / main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            if self.loss == "pair":
                self.valid()
            else:
                self.score()
            self.save()

    @torch.no_grad()
    def score(self):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        for graph_pair in tqdm(self.testing_graphs):
            pair_gen = process_pair(graph_pair)
            while True:
                try:
                    data = next(pair_gen)
                    self.ground_truth.append(calculate_normalized_ged(data))
                    data = self.transfer_to_torch(data)
                    data = assign_GPU(data, self.device)
                    target = data["target"]
                    prediction = self.model(data)
                    self.scores.append(calculate_loss(prediction, target))
                except:
                    break
        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n - norm_ged_mean) ** 2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nBaseline error: " + str(round(base_error, 5)) + ".")
        print("\nModel test error: " + str(round(model_error, 5)) + ".")

    def save(self):
        torch.save(self.model.state_dict(), self.args.save_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.args.load_path))

    def train(self, dataloader):
        model.train()
        for batch in dataloader:
            pass

    @torch.no_grad()
    def valid(self):
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.scores = []
        for graph_pair in tqdm(self.testing_graphs):
            pair_gen = process_pair(graph_pair)
            while True:
                try:
                    data = next(pair_gen)
                    data = self.transfer_to_torch(data)
                    data = assign_GPU(data, self.device)
                    target = data["target"]
                    prediction = self.model(data)
                    pair_loss = pairwise_loss(
                        prediction[0].view(1, -1),
                        prediction[1].view(1, -1),
                        data["target"],
                        loss_type="margin",
                        margin=1.0,
                    )
                    self.scores.append(pair_loss.detach().cpu().numpy())
                except:
                    break

        model_error = np.mean(self.scores)
        print("\nModel test loss: " + str(round(model_error, 5)) + ".")
