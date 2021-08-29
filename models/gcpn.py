from torchdrug import core, models, tasks
from torch import nn, optim
import pickle

with open("zinc250k.pkl", "rb") as fin:
  dataset = pickle.load(fin)


model = models.RGCN(input_dim=dataset.node_feature_dim,
                    num_relation=dataset.num_bond_type,
                    hidden_dims=[256, 256, 256, 256], batch_norm=False)
task = tasks.GCPNGeneration(model, dataset.atom_types, max_edge_unroll=12,
                            max_node=38, criterion="nll")

optimizer = optim.Adam(task.parameters(), lr = 1e-3)
#solver = core.Engine(task, dataset, None, None, optimizer,
#                     gpus=(0,), batch_size=128, log_interval=10)

solver = core.Engine(task, dataset, None, None, optimizer,
                     batch_size=128, log_interval=10)

epochs = 1
solver.train(num_epoch=epochs)
solver.save("gcpn_zinc250k_1" + str(epochs) + ".pkl")                        
