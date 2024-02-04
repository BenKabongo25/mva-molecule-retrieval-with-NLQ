!pip install -r requirements.txt

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

from tqfm import tqdm

from alignment import AlignmentModel,Discriminator, gradient_penalty, CombinedModel
# from moemodel import MOEModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score

from losses import hard_triplet_loss, expo_triplet_loss, info_nce_loss, contrastive_weighted_loss, cosine_contrastive_weighted_loss, sigmoid_loss


batch_size = 64
sub_batch_size = 32

learning_rate = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'allenai/scibert_scivocab_uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]

val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model = AlignmentModel(in_channels=300, out_channels=768, graph_attention_head=6, type = 'TransformerConv', n_layers = 5)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999),
                                weight_decay=0.01)
model.to(device)

epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 100
best_validation_loss = 1000000
best_validation_mrr = 0

sub_batch_size = 32
nb_epochs = 30
loss = 0
beta = 1.0

for e in range(nb_epochs):
    print('-----EPOCH{}-----'.format(e+1))
    model.train()
    
    for batch in train_loader:
        input_ids = batch.input_ids.to(device)
        batch.pop('input_ids')
        attention_mask = batch.attention_mask.to(device)
        batch.pop('attention_mask')
        batch.to(device)
        
        acc = 0
        partial_loss = 0
        
        
    
        for i in range(0, len(batch.ptr) -1 , sub_batch_size):
    
            f_index_i = min(i+ sub_batch_size, len(batch.ptr) - 1)
    
            sub_batch = batch.batch[batch.ptr[i]: batch.ptr[f_index_i]]
            
            sub_batch = sub_batch - sub_batch[0]
            
            sub_x = batch.x[batch.ptr[i]: batch.ptr[f_index_i]]
            
            max_index = batch.ptr[f_index_i]
            
            min_index = batch.ptr[i]
            
            sub_edge_index = torch.stack(
                                            (
                                                batch.edge_index[0][(min_index <= batch.edge_index[0]) & (batch.edge_index[0] < max_index)],
                                                batch.edge_index[1][(min_index <= batch.edge_index[1]) & (batch.edge_index[1] < max_index)]
                                            ),
                                            dim=0
                                        )
            
            sub_edge_index = sub_edge_index - min_index        
            
            # sub_graph_batch = Data(x = sub_x, edge_index = sub_edge_index, batch = sub_batch).to(device)
    
            graph_embeddings = model.forward_graph(sub_x, sub_edge_index, sub_batch)   
    
            cosine = torch.zeros(len(input_ids), graph_embeddings.shape[0]).to(device)
        
            for j in range(0, len(input_ids), sub_batch_size):
                
                # sub_input_ids = input_ids[j : j + sub_batch_size]
                
                # sub_attention_mask = attention_mask[j : j + sub_batch_size]
    
                text_embeddings = model.forward_text(input_ids[j : j + sub_batch_size], 
                                attention_mask[j : j + sub_batch_size])
                
                sub_cosine = pairwise_cosine_similarity(text_embeddings, graph_embeddings)
    
                cosine[j:j + text_embeddings.shape[0]] = sub_cosine
                
            current_loss = cosine_contrastive_weighted_loss(cosine,  pos = i, dim =0)
            current_loss.backward()
            acc += 1
            partial_loss += current_loss.item()
            
    
        for j in range(0, len(input_ids), sub_batch_size):
                
            sub_input_ids = input_ids[j : j + sub_batch_size]
            
            sub_attention_mask = attention_mask[j : j + sub_batch_size]
    
            text_embeddings = model.forward_text(sub_input_ids, 
                            sub_attention_mask)
            
            cosine = torch.zeros(text_embeddings.shape[0], len(batch.ptr) - 1).to(device)
            
            for i in range(0, len(batch.ptr) -1 , sub_batch_size):
        
                f_index_i = min(i+ sub_batch_size, len(batch.ptr) - 1)
        
                sub_batch = batch.batch[batch.ptr[i]: batch.ptr[f_index_i]]
                
                sub_batch = sub_batch - sub_batch[0]
                
                sub_x = batch.x[batch.ptr[i]: batch.ptr[f_index_i]]
                
                max_index = batch.ptr[f_index_i]
                
                min_index = batch.ptr[i]
                
                sub_edge_index = torch.stack(
                                                (
                                                    batch.edge_index[0][(min_index <= batch.edge_index[0]) & (batch.edge_index[0] < max_index)],
                                                    batch.edge_index[1][(min_index <= batch.edge_index[1]) & (batch.edge_index[1] < max_index)]
                                                ),
                                                dim=0
                                            )
                
                sub_edge_index = sub_edge_index - min_index        
                
                # sub_graph_batch = Data(x = sub_x, edge_index = sub_edge_index, batch = sub_batch).to(device)
        
                graph_embeddings = model.forward_graph(sub_x, sub_edge_index, sub_batch)       
                
                sub_cosine = pairwise_cosine_similarity(text_embeddings, graph_embeddings)
    
                cosine[:,i:i + graph_embeddings.shape[0]] = sub_cosine
                
            current_loss = cosine_contrastive_weighted_loss(cosine, pos = j, dim =1)
            current_loss.backward()
            acc += 1
            partial_loss += current_loss.item()

        for param in model.parameters():
            if param.grad is not None:
                param.grad /= acc
                    
        optimizer.step()

        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
            
        loss += partial_loss
        partial_loss = 0
            
        count_iter += 1
        
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0
            
    model.eval()       
    val_loss = 0   
    graphs = []
    texts = []     
    for batch in val_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        graph_embeddings = model.forward_graph(x = batch.x.to(device), edge_index = batch.edge_index.to(device), batch = batch.batch.to(device)) 
        text_embeddings = model.forward_text(input_ids.to(device), 
                                attention_mask.to(device))
        graphs.extend(graph_embeddings.tolist())
        texts.extend(text_embeddings.tolist())

    best_validation_loss = min(best_validation_loss, val_loss)
    print('-----EPOCH'+str(e+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )
    similarity = cosine_similarity(texts, graphs)
    y_true = np.eye(len(similarity))
    score = label_ranking_average_precision_score(y_true, similarity)
    print('-----EPOCH'+str(e+1)+'----- done.  Validation MRR: ', str(score) )
    best_validation_mrr = max(best_validation_mrr, score)
    if best_validation_mrr==score:
        current_directory = os.getcwd()
        files = os.listdir(current_directory)

        for file in files:
            if file.startswith('model'):
                file_path = os.path.join(current_directory, file)
                os.remove(file_path)
                
        print('validation loss improoved saving checkpoint...')
        save_path = os.path.join('./', 'model'+str(e)+'.pt')
        torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_accuracy': val_loss,
        'MRR' : score,
        'loss': loss,
        }, save_path)
        print('checkpoint saved to: {}'.format(save_path))

    if e and e%15 == 0:
        optimizer.param_groups[0]['lr'] /= 10
    
        
current_directory = os.getcwd()
files = os.listdir(current_directory)

for file in files:
    if file.startswith('model'):  
        save_path = os.path.join(current_directory, file)
        
print('loading best model...')
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

graph = []
text= []

test_graph_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

for batch in tqdm(test_graph_loader):
    output = model.forward_graph(x = batch.x.to(device), edge_index = batch.edge_index.to(device), batch = batch.batch.to(device)) 
    graph.extend(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)

for batch in tqdm(test_text_loader):

    output = model.forward_text(batch['input_ids'].to(device), batch['attention_mask'].to(device))
    text.extend(output.tolist())


similarity = cosine_similarity(text, graph)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv('submission.csv', index=False)    
