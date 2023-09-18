from evaluation import compute_similarity, auc
from loss import pairwise_loss, triplet_loss
from utils import *
from configure import *
import numpy as np
import torch.nn as nn
import collections
import time
import os

# dot_product
def pairwise_dot_product_similarity(x, y):
    num = np.dot(x, np.array(y).T)  # 向量点乘
    return num


# cosine sim
def pairwise_cosine_similarity(x, y):
    num = np.dot(x, np.array(y).T)  # 向量点乘
    denom = np.linalg.norm(x) * np.linalg.norm(y, axis=1)  # 求模长的乘积
    num = num / denom
    return num


def euclidean__similarity(x, y):
    """This is the squared Euclidean distance."""
    scores = -torch.sum((x - y) ** 2, dim=-1)
    scores_max = torch.max(scores)
    scores_min = torch.min(scores)

    # normalize scores to [0, 1] and add a small epislon for safety
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)
    return scores


def test(batch_size=2):
    model.eval()
    with torch.no_grad():
        for i in range(2):
            batch_fid, batch = next(test_data_iter)
            (node_features, edge_features, from_idx, to_idx, graph_idx) = get_graph(
                batch
            )

            eval_pairs = model(
                node_features.to(device),
                edge_features.to(device),
                from_idx.to(device),
                to_idx.to(device),
                graph_idx.to(device),
                graph_idx.max() + 1,  # batch_size,
            )

            print(eval_pairs.shape)
            for i in eval_pairs:
                print(euclidean__similarity(i.view(1, -1), eval_pairs))


# Set GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Print configure
config = get_default_config()
for (k, v) in config.items():
    print("%s= %s" % (k, v))

batch_size = 32  # config["evaluation"]["batch_size"]

# Set random seeds
seed = config["seed"]
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# load data
test_set = build_test_datasets(config)
test_data_iter = test_set.single(batch_size=batch_size)

# load model
model_file = "checkpoint/embed_model.pt"
model = torch.load(model_file).to(device)

# print(model)

# f = open("test_result.txt", "w")

t_start = time.time()
test(batch_size=batch_size)

# f.write(f"{sim}\n")

# if i_iter % 100 == 0:
#     print("iter %d, time %.2fs" % (i_iter + 1, time.time() - t_start))

# f.close()
