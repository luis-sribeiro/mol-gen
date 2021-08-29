import torch
from torchdrug import datasets
import pickle

dataset = datasets.ZINC250k("~/molecule-datasets/", kekulize=True,
                            node_feature="symbol")

with open("zinc250k.pkl", "wb") as fout:
  pickle.dump(dataset, fout)