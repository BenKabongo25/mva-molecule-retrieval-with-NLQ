import os
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GCNConv, GINConv, GPSConv, GINEConv, EGConv, AntiSymmetricConv, SuperGATConv, GATv2Conv,DirGNNConv

from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer




from transformers import AutoModel

import torch.nn.functional as F


ce_criterion = nn.CrossEntropyLoss()


def gradient_penalty(discriminator, graph_embeddings, text_embeddings):

    device = graph_embeddings.device
    alpha = torch.rand((text_embeddings.shape[0], 1)).to(device)

    interpolated = graph_embeddings * alpha + text_embeddings * (1 - alpha)

    # Calculate critic scores
    output = discriminator(interpolated)
    
    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=output,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

class Discriminator(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1),
            nn.Sigmoid())

    def forward(self, x):
        return self.model(x)

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, graph_attention_head, type = 'GPS', n_layers = 5, concat=False, beta=True):
        super(GraphEncoder, self).__init__()

        self.gnns = nn.ModuleList([])
              
        self.encoder_type = type
       
        self.projection = nn.Linear(in_channels, out_channels)
        
        for i in range(n_layers):
            self.gnns.append(self.get_gnn(out_channels, out_channels, graph_attention_head, type))
            
        self.graph_norms = nn.ModuleList([nn.LayerNorm((out_channels)) for _ in range(n_layers)])
    
        self.graph_forward = nn.Sequential(nn.Linear(out_channels, 2*out_channels), nn.ReLU(), nn.Linear(2*out_channels, out_channels))
        
    def get_gnn(self, in_channels, out_channels, graph_attention_head, type = 'GPS', concat = False ):
        if type == 'TransformerConv':
            return TransformerConv(in_channels=in_channels, out_channels=out_channels, heads=graph_attention_head, concat=concat, beta=True)

        elif type == 'GPS':
            return GPSConv(channels=in_channels,conv = GINConv(nn.Sequential(nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Linear(out_channels, out_channels))),heads=graph_attention_head,dropout=0.1)
            
        elif type == 'GIN':
            return GINConv(nn.Sequential(nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Linear(out_channels, out_channels)))
            
        elif type == 'EGC':
            return EGConv(in_channels=in_channels, out_channels=out_channels, aggregators=["sum", "mean", "symnorm", "max"], num_heads = 6, num_bases=6, add_self_loops=False)
            
        elif type == 'Antisymmetric':
            return AntiSymmetricConv(in_channels=in_channels, phi = GINConv(nn.Sequential(nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Linear(out_channels, in_channels))), num_iters = 1, act = 'relu')
            
        elif type == 'SuperGat':
            return SuperGATConv(in_channels=in_channels,out_channels=out_channels, heads=graph_attention_head, concat=concat, add_self_loops = True)

        elif type == 'GATv2Conv':
            return GATv2Conv(in_channels=in_channels,out_channels=out_channels, heads=graph_attention_head, concat=concat, add_self_loops = True)
            
        elif type == 'Antisymmetric_supergat':
            return AntiSymmetricConv(in_channels=in_channels, phi = SuperGATConv(in_channels=in_channels,out_channels=out_channels, heads=graph_attention_head, concat=concat, add_self_loops = True), num_iters = 1, act = 'relu')

        elif type == 'PDNConv':
            return PDNConv(in_channels=in_channels,out_channels=out_channels, edge_dim = 300,  hidden_channels = 512)
            
        elif type == 'DirGNNConv_supergat':
            return DirGNNConv(SuperGATConv(in_channels=in_channels,out_channels=out_channels, heads=graph_attention_head, concat=concat, add_self_loops = True))

        elif type == 'DirGNNConv_trans':
            return DirGNNConv(TransformerConv(in_channels=in_channels, out_channels=out_channels, heads=graph_attention_head, concat=concat, beta=True))
            
        
        


    def forward(self, x, edge_index, batch, return_node_feats = False):
    
        node_feat = x

        node_feat = self.projection(node_feat)
        
        if len(edge_index) != 0:
            for i,conv in enumerate(self.gnns):
                node_feat = conv(node_feat, edge_index)
                node_feat = node_feat.relu()
                node_feat = self.graph_norms[i](node_feat)                
                
        graph_output = global_add_pool(node_feat, batch)

        graph_output = self.graph_forward(graph_output)

        if return_node_feats:
            return graph_output, node_feat
        else:
            return graph_output

class TextEncoder(nn.Module):
    def __init__(self, out_channels, load_pretrained = True, dim = 768, pretrained_path = 'text_encoder_pretrained.pth'):
        super(TextEncoder, self).__init__()

        if 'roberta' in pretrained_path:
            self.text_encoder = AutoModel.from_pretrained('allenai/biomed_roberta_base')
            self.type = 'roberta'
        elif 'deberta' in pretrained_path:
            self.text_encoder = AutoModel.from_pretrained('KISTI-AI/scideberta-cs')
            self.type = 'deberta' 
        elif 'llama' in pretrained_path:
            self.text_encoder = AutoModel.from_pretrained('OpenBioMed/Med-LLaMA-7b')
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
            lora_config = LoraConfig(
                r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none"
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)
            self.type = 'llama'
        else:
            self.text_encoder = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
            self.type = 'bert'
        
            if load_pretrained:
                if 'kv_plm' in pretrained_path:
                    print('-'*20 + 'loading pretrained' + '-'*20)
                    load_dict = torch.load(pretrained_path)
                    suffix = 'main_model.'
                    load_dict = {k[len(suffix):]:v for k,v in load_dict.items()}
                    
                else:
                    print('-'*20 + 'loading pretrained' + '-'*20)
                    load_dict = torch.load(pretrained_path)
                self.text_encoder.load_state_dict(load_dict)
        self.text_forward = nn.Sequential(nn.Linear(dim, 2*out_channels), nn.ReLU(), nn.Linear(2*out_channels, out_channels))
        self.text_norm = nn.LayerNorm((dim))
    
    def forward(self, text: Tensor, text_mask: Tensor, return_seq_feats = False) -> Tensor:
        
        features = self.text_encoder(text, text_mask).last_hidden_state

        if return_seq_feats:
            features = self.text_norm(features)
            features = self.text_forward(features)
            return features
        
        if self.type == 'bert':
            text_output = features[:,0,:]
        elif self.type in ['roberta', 'deberta']:
            text_output = features * text_mask.unsqueeze(2)
            text_output = torch.sum(text_output, dim = 1)/torch.sum(text_mask, dim = 1, keepdim = True)
        elif self.type == 'llama':
            text_output = features[:,-1,:]
            
            
        text_output = self.text_norm(text_output)
        text_output = self.text_forward(text_output)

        return text_output
    


class AlignmentModel(nn.Module):
    def __init__(self, in_channels, out_channels, graph_attention_head, type = 'GPS', n_layers = 5, concat=False, beta=True, dim = 768, temp = None, t_prime = None, b= None,  load_pretrained = True, pretrained_path = 'text_encoder_pretrained.pth'):
        super(AlignmentModel, self).__init__()

        # Text Encoder

        if temp is not None:
            self.temp = nn.Parameter(torch.Tensor([temp]))
        if t_prime is not None:
            self.t_prime = nn.Parameter(torch.Tensor([t_prime]))
        if b is not None:
            self.b = nn.Parameter(torch.Tensor([b]))
            
        
        self.text_encoder = TextEncoder(out_channels, load_pretrained , dim , pretrained_path )
        self.graph_encoder = GraphEncoder(in_channels, out_channels, graph_attention_head, type , n_layers, concat, beta)

    def forward(self, graph_batch, text: Tensor, text_mask: Tensor):
        
        return self.graph_encoder(graph_batch), self.text_encoder(text, text_mask)

    def forward_text(self, text: Tensor, text_mask: Tensor) -> Tensor:

        return self.text_encoder(text, text_mask)

    def forward_graph(self, x, edge_index, batch, return_node_feats = False):

        return self.graph_encoder(x, edge_index, batch, return_node_feats)

