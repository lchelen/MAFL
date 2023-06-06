import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

def update_clients_weights_by_uncertainty(_clients_weights, args):
    update_factor = args.update_factor
    updated_clients_weights = [(1 - update_factor)*j for j in zip(_clients_weights)]
    print(updated_clients_weights)
    return updated_clients_weights

def update_global_model(net_clients, client_weight, args):
    #meger the weights to centre model and assign to each client
    for param in zip(net_clients[0].parameters(), net_clients[1].parameters(),
                     net_clients[2].parameters()):
        #merge
        if param[0].shape != param[1].shape:
            continue
        new_para = torch.tensor(np.zeros(param[0].shape), requires_grad=False)
        for i in range(args.client_num):
            new_para.data.add_(param[i].data, alpha=client_weight[i])
        #assign
        for i in range(args.client_num):
            param[i].data.mul_(0).add_(new_para.data)

def euclidean(matrix1, matrix2):
    euclidean_dist = 1 / (1 + np.sqrt(np.square(matrix1 - matrix2).sum()))
    return euclidean_dist

def Federated_update(net_clients, args):
    for param in zip(net_clients[0].parameters(), net_clients[1].parameters(), net_clients[2].parameters()):
        if param[0].shape != param[1].shape:
            continue
        for i in range(args.client_num):
            if i == 0:
                compare1 = param[i + 1]
                compare2 = param[i + 2]
            elif i == 1:
                compare1 = param[i - 1]
                compare2 = param[i + 1]
            elif i == 2:
                compare1 = param[i - 2]
                compare2 = param[i - 1]
            score1 = euclidean(param[i].detach().numpy(), compare1.detach().numpy())
            score2 = euclidean(param[i].detach().numpy(), compare2.detach().numpy())
            if score1 >= args.threshold and score2 >= args.threshold:
                param[i].data.add_(compare1.data, alpha = 1/3)
                param[i].data.add_(compare2.data, alpha = 1/3)
            elif score1 >= args.threshold and score2 < args.threshold:
                param[i].data.add_(compare1.data, alpha = 1/2)
            elif score1 < args.threshold and score2 >= args.threshold:
                param[i].data.add_(compare2.data, alpha = 1/2)
