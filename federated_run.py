import numpy as np
from multipleParser import get_parser
import torch
import random
from torch.autograd import Variable
from datetime import datetime
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

args = get_parser().parse_args()
set_random_seed(args.seed)

import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import load_data
import torch.nn.functional as F
#from evaluation_metrics import eval_data
from F_GCRN import F_GCRN
from Federated import update_clients_weights_by_uncertainty, update_global_model, Federated_update

path = '...//Federated//'
file1 = path + 'NY_bike_1h.csv'
file2 = path + 'NY_taxi_1h.csv'
file3 = path + 'BJ_taxi_1h.csv'
file = [file1, file2, file3]

data, label, train_data, train_label, validate_data, validate_label, test_data, test_label, \
sc = load_data.load_for_federated(file, args.lag)

print('Data load finish!')

criterion = nn.MSELoss(size_average=False)

model_clients = []
optimizer_clients = []

for client_idx in range(len(file)):
    args.rnn_units = 64
    args.num_layers = 2
    args.embed_dim = 2
    args.cheb_k = 3
    args.num_nodes = data[client_idx].shape[2]
    model = AGCRN(args)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    model_clients.append(model)
    optimizer_clients.append(optimizer)
    
clients_weights = [1.0 / len(file) for i in range(len(file))]

for epoch in range(1, args.epochs + 1):
    for client_idx in range(len(file)):
        dataloader_current = data[client_idx]
        model_current = model_clients[client_idx]
        optimizer_current = optimizer_clients[client_idx]
        train_times = int(train_data[client_idx].shape[0] / args.batch_size)
        
        model_current.train()
        train_loss = 0
        
        for i in range(train_times + 1):
            if i < train_times:
                batch_data = train_data[client_idx][i * args.batch_size : i * args.batch_size + args.batch_size, :, :, :].float()
                batch_label = train_label[client_idx][i * args.batch_size:i * args.batch_size + args.batch_size, :].float()
            else:
                batch_data = train_data[client_idx][train_times * args.batch_size:, :, :, :].float()
                batch_label = train_label[client_idx][train_times * args.batch_size :].float()
            pred = model_current(batch_data).squeeze()
            loss = criterion(batch_label, pred)
            optimizer_current.zero_grad()
            loss.backward()
            optimizer_current.step()
            train_loss = train_loss + loss.item() * batch_label.shape[0]
        train_loss = train_loss / train_data.shape[0]

    Federated_update(model_clients, args)
