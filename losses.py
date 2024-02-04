import time
import os
import pandas as pd
import numpy as np

from dataloader import GraphTextDataset, GraphDataset, TextDataset

import torch
import torch.nn as nn
from torch import optim
from torch_geometric.data import DataLoader, Data
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from torchmetrics.functional import pairwise_cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score

ce_criterion = nn.CrossEntropyLoss()
logsigmoid = nn.LogSigmoid()
CE = torch.nn.CrossEntropyLoss()


def hard_triplet_loss(graph_embeddings, text_embeddings, margin = 0.3):
    cosine = pairwise_cosine_similarity(text_embeddings, graph_embeddings) # compute cosine similarity between each pairs (texts, graphs)
    positive_sample = cosine.diag() # get similarity between anchor and positive sample where anchor could be the text representation and positive sample the graph represention and vice versa
    cosine = cosine.fill_diagonal_(-2) # set diag val to a minimum possible value of similarity to get hard negetive example by argmax
    loss = torch.clamp(torch.max(cosine, axis = 1)[0] - positive_sample + margin,0)
    loss += torch.clamp(torch.max(cosine, axis = 0)[0] - positive_sample +  margin,0)
    loss = torch.mean(loss)
    return loss
    
def expo_triplet_loss(graph_embeddings, text_embeddings, margin = 0.3):
    cosine = pairwise_cosine_similarity(text_embeddings, graph_embeddings)
    scores = torch.exp(cosine - torch.diag(cosine).reshape(-1,1) + margin)
    scores = scores - torch.diag(torch.diag(scores))
    loss = torch.log(torch.max(scores, dim = 0)[0] + torch.max(scores, dim = 1)[0])
    loss = loss.relu()
    loss = loss ** 2
    loss = torch.mean(loss)
    return loss

def info_nce_loss(graph_embeddings, text_embeddings, temp = 0.1):
    cosine = pairwise_cosine_similarity(text_embeddings, graph_embeddings)/temp
    labels = torch.arange(cosine.shape[0], device=cosine.device)

    return CE(cosine, labels) + CE(torch.transpose(cosine, 0, 1), labels)
    
def contrastive_weighted_loss(graph_embeddings, text_embeddings, beta = 1.0, temp = 0.1):
    cosine = pairwise_cosine_similarity(text_embeddings, graph_embeddings)/temp
    cosine = torch.exp(cosine)
    positives = cosine.diag()

    num_neg = cosine.shape[0] - 1
    
    cosine = cosine*((torch.ones(cosine.shape) - torch.eye(cosine.shape[0])).to(cosine.device))
    
    negatives = cosine *  (cosine/torch.sum(cosine, dim = 1 , keepdim = True))
    negatives = torch.sum(negatives, dim = 1)
    loss = -torch.log(positives/(positives + beta*num_neg*negatives))

    negatives = cosine *  (cosine/torch.sum(cosine, dim = 0 , keepdim = True))
    negatives = torch.sum(negatives, dim = 0)
    loss += -torch.log(positives/(positives + beta*num_neg*negatives))

    return torch.mean(loss)   

def cosine_contrastive_weighted_loss(cosine, pos , beta = 1.0, temp = 0.1, dim = 1):
    cosine = cosine/temp
    cosine = torch.exp(cosine)
    

    num_neg = cosine.shape[dim] - 1

    mask = torch.ones(cosine.shape)
    r = min(mask.shape)

    if dim == 0:
        positives = cosine[pos: pos + r].diag()
        
        if pos + r > len(mask):
            row = torch.arange(pos ,min(pos + r, len(mask))) 
            col = torch.arange(pos ,min(pos + r, len(mask))) - pos
        else:
            row = pos + torch.arange(r)
            col = torch.arange(r)
            
        mask[row, col] = 0
        
    if dim == 1:
        
        positives = cosine[:,pos : pos + r].diag()

        if pos + r > len(mask):
            col = torch.arange(pos ,min(pos + r, len(mask[0]))) 
            row = torch.arange(pos ,min(pos + r, len(mask[0]))) - pos
        else:
            col = pos + torch.arange(r)
            row = torch.arange(r)
        
        mask[row, col] = 0   
        
    
    cosine = cosine*mask.to(cosine.device)
    
    negatives = cosine *  (cosine/torch.sum(cosine, dim, keepdim = True))
    negatives = torch.sum(negatives, dim)
    loss = -torch.log(positives/(positives + beta*num_neg*negatives))
    
    return torch.mean(loss) 

    
        

def sigmoid_loss(text_embeddings, graph_embeddings, t_prime, b):
    cosine = pairwise_cosine_similarity(text_embeddings, graph_embeddings)
    
    batch_size = cosine.shape[0]
    
    positives_samples = logsigmoid(torch.exp(t_prime)*cosine.diag() - b)
    
    cosine = cosine.fill_diagonal_(-2)
    
    negatives_samples = logsigmoid(-torch.exp(t_prime)* torch.max(cosine, axis = 1)[0] + b) +  logsigmoid(-torch.exp(t_prime)*torch.max(cosine, axis = 0)[0] + b) 
    
    return -(torch.mean(positives_samples) + torch.mean(negatives_samples))
    