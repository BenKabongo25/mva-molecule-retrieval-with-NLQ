import os

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GCNConv, GINConv, GPSConv, GINEConv, EGConv, AntiSymmetricConv

from transformers import AutoModel


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

class AlignmentModel(nn.Module):
    def __init__(self, in_channels, out_channels, graph_attention_head, type = 'GPS', concat=False, beta=True, load_pretrained = True, pretrained_path = 'text_encoder_pretrained.pth'):
        super(AlignmentModel, self).__init__()

        # Text Encoder
        self.text_encoder = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        if load_pretrained and os.path.exists(pretrained_path):
            self.text_encoder.load_state_dict(torch.load(pretrained_path))
        self.text_forward = nn.Sequential(nn.Linear(768, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
        self.text_norm = nn.LayerNorm((768))


        #Graph Encoder
        if type == 'TransformerConv':
            self.conv1 = TransformerConv(in_channels=in_channels, out_channels=out_channels, heads=graph_attention_head, concat=concat, beta=beta)
            self.conv2 = TransformerConv(in_channels=out_channels, out_channels=out_channels, heads=graph_attention_head, concat=concat, beta=beta)
            self.conv3 = TransformerConv(in_channels=out_channels, out_channels=out_channels, heads=graph_attention_head, concat=concat, beta=beta)

        elif type == 'GPS':
            self.conv1 = GPSConv(channels=in_channels,conv = GINConv(nn.Sequential(nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Linear(out_channels, out_channels))),heads=graph_attention_head,dropout=0.1)
            self.conv2 = GPSConv(channels=in_channels,conv = GINConv(nn.Sequential(nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Linear(out_channels, out_channels))),heads=graph_attention_head,dropout=0.1)
            self.conv3 = GPSConv(channels=in_channels,conv = GINConv(nn.Sequential(nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Linear(out_channels, out_channels))),heads=graph_attention_head,dropout=0.1)

        elif type == 'GIN':
            self.conv1 = GINConv(nn.Sequential(nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Linear(out_channels, out_channels)))
            self.conv2 = GINConv(nn.Sequential(nn.Linear(out_channels, out_channels),nn.ReLU(),nn.Linear(out_channels, out_channels)))
            self.conv3 = GINConv(nn.Sequential(nn.Linear(out_channels, out_channels),nn.ReLU(),nn.Linear(out_channels, out_channels)))

        elif type == 'EGC':
            self.conv1 = EGConv(in_channels=in_channels, out_channels=out_channels, aggregators=["sum", "mean", "symnorm", "max"], num_heads = 6, num_bases=6, add_self_loops=False)
            self.conv2 = EGConv(in_channels=in_channels, out_channels=out_channels, aggregators=["sum", "mean", "symnorm", "max"], num_heads = 6, num_bases=6, add_self_loops=False)
            self.conv3 = EGConv(in_channels=in_channels, out_channels=out_channels, aggregators=["sum", "mean", "symnorm", "max"], num_heads = 6, num_bases=6, add_self_loops=False)
            
        elif type == 'Antisymmetric':
            self.conv1 = AntiSymmetricConv(in_channels=in_channels, phi = GINConv(nn.Sequential(nn.Linear(in_channels, out_channels),nn.ReLU(),nn.Linear(out_channels, out_channels))), num_iters = 1, act = 'relu')
            self.conv2 = AntiSymmetricConv(in_channels=out_channels, phi = GINConv(nn.Sequential(nn.Linear(out_channels, out_channels),nn.ReLU(),nn.Linear(out_channels, out_channels))), num_iters = 1, act = 'relu')
            self.conv3 = AntiSymmetricConv(in_channels=out_channels, phi = GINConv(nn.Sequential(nn.Linear(out_channels, out_channels),nn.ReLU(),nn.Linear(out_channels, out_channels))), num_iters = 1, act = 'relu')

        self.graph_forward = nn.Sequential(nn.Linear(out_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
        self.graph_norm = nn.LayerNorm((out_channels))


    def forward(self, graph_batch, text: Tensor, text_mask: Tensor):
        """
        Performs forward propagation using both graph and text inputs.

        Args:
            graph_batch (Batch): The batch of graph data.
            text (Tensor): The tensor representing the input text.
            text_mask (Tensor): The tensor representing the text mask.

        Returns:
            Tuple[Tensor, Tensor]: The resulting tensors after forward propagation for graph and text.
        """
        
        text_output = self.text_encoder(text, text_mask).last_hidden_state[:,0,:]
        text_output = self.text_norm(text_output)
        text_output = self.text_forward(text_output)

        edge_index = graph_batch.edge_index
        graph_output = self.conv1(graph_batch.x, edge_index)
        graph_output = graph_output.relu()

        graph_output = self.conv2(graph_output, edge_index)
        graph_output = graph_output.relu()

        graph_output = self.conv3(graph_output, edge_index)
        graph_output = global_add_pool(graph_output, graph_batch.batch)

        graph_output = self.graph_norm(graph_output)
        graph_output = self.graph_forward(graph_output)

        return graph_output, text_output

    def forward_text(self, text: Tensor, text_mask: Tensor) -> Tensor:
        """
        Performs forward propagation using only the text.

        Args:
            text (Tensor): The tensor representing the input text.
            text_mask (Tensor): The tensor representing the text mask.

        Returns:
            Tensor: The resulting tensor after forward propagation.
        """
        
        text_output = self.text_encoder(text, text_mask).last_hidden_state[:,0,:]
        text_output = self.text_norm(text_output)
        text_output = self.text_forward(text_output)

        return text_output

    def forward_graph(self, graph_batch):
        """
        Performs forward propagation using a graph batch.

        Args:
            graph_batch (Batch): The batch of graph data.

        Returns:
            Tensor: The resulting tensor after forward propagation.
        """

        edge_index = graph_batch.edge_index
        graph_output = self.conv1(graph_batch.x, edge_index)
        graph_output = graph_output.relu()

        graph_output = self.conv2(graph_output, edge_index)
        graph_output = graph_output.relu()

        graph_output = self.conv3(graph_output, edge_index)
        graph_output = global_add_pool(graph_output, graph_batch.batch)

        graph_output = self.graph_norm(graph_output)
        graph_output = self.graph_forward(graph_output)

        return graph_output