from torchdrug import core, models, tasks, datasets
from torch import nn, optim
import torch
import pickle

download = True

dataset = None

if download == True:
  dataset = datasets.ZINC250k("~/molecule-datasets/", kekulize=True,
                            node_feature="symbol")
  with open("zinc250k.pkl", "wb") as fout:
    pickle.dump(dataset, fout)
else:
  with open("zinc250k.pkl", "rb") as fin:
    dataset = pickle.load(fin)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpus = None

if str(device) == 'cuda':
  print('Available GPU: ', torch.cuda.get_device_name(device))
  gpus = (0,)
else:
  print('There are no GPUs available')
  gpus = None

model = models.RGCN(input_dim=dataset.node_feature_dim,
                    num_relation=dataset.num_bond_type,
                    hidden_dims=[256, 256, 256, 256], batch_norm=False)

model = model.to(device)

task = tasks.GCPNGeneration(model, dataset.atom_types, max_edge_unroll=12,
                            max_node=38, criterion="nll")

optimizer = optim.Adam(task.parameters(), lr = 1e-3)


solver = core.Engine(task, dataset, None, None, optimizer,
                     gpus=gpus, batch_size=128, log_interval=10)

epochs = 10
print('Pre-training the gcpn model for {} epochs'.format(epochs))
for epoch in range(epochs):
  solver.train(num_epoch=1)
  solver.save("gcpn_zinc250k_" + str(epoch) + ".pkl")  
  print('Model saved!')    