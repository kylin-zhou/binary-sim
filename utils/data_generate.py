import json
import pickle
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import PairDataset
from utils import ConfigParse

config_file = json.load(open("config.json", "r"))
config = ConfigParse(config_file)

dataset = PairDataset(config)

print("start!")
all_pairs = []
for i in tqdm(range(len(dataset))):
    pair = dataset.__getitem__(i)
    all_pairs.append(pair)

with open(config.processed_file, "wb") as f:
    pickle.dump(all_pairs, f, protocol=4)

print("Finished!")
