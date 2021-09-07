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
  print('There are not GPUs available')
  gpus = None

model = models.RGCN(input_dim=dataset.node_feature_dim,
                    num_relation=dataset.num_bond_type,
                    hidden_dims=[256, 256, 256, 256], batch_norm=False)

model = model.to(device)

task = tasks.GCPNGeneration(model, dataset.atom_types,
                            max_edge_unroll=12, max_node=38,
                            task="qed", criterion=("ppo", "nll"),
                            reward_temperature=1,
                            agent_update_interval=3, gamma=0.9)

optimizer = optim.Adam(task.parameters(), lr=1e-5)

solver = core.Engine(task, dataset, None, None, optimizer,
                     gpus=gpus, batch_size=16, log_interval=10)


solver.load("gcpn_zinc250k_8.pkl", load_optimizer=False)
epochs = 10
for epoch in range(epochs):
  solver.train(num_epoch=1)
  solver.save("gcpn_zinc250k_finetune_" + str(epoch) + ".pkl")  
  print('Model saved!')                 

results = task.generate(num_sample=32, max_resample=5)
print(results.to_smiles())