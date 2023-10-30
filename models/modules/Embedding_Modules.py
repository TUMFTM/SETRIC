import os
import sys

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_path)

import torch
import torch.nn as nn
from models.modules.module_utils import CustomModule, CustomConv


class GNN_Embedding_CustomConv(CustomModule):
    def __init__(
        self,
        input_features,
        message_size,
        aggr,
        embedding_size,
        num_hidden_layers,
        hidden_size,
    ):
        super(GNN_Embedding_CustomConv, self).__init__()

        """
        Module description: 
        The GNN_Embedding Module transforms a node feature vector (of dimension input_features) into a node feature 
        embedding (of dimension embedding_size) via multiple graph convolution layers. The forward method takes takes 
        the graph node feature tensor and edge_index tensor and outputs an embedded node feature tensor.
        """

        self.input_features = input_features
        self.message_size = message_size
        self.aggr = aggr
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

        self.gnn_conv_in = CustomConv(
            in_dim=self.input_features,
            message_dim=self.message_size,
            out_dim=self.hidden_size,
            aggr=aggr,
            message_coord_transformation=True,
        )
        self.gnn_conv_hidden = nn.ModuleList(
            [
                CustomConv(
                    in_dim=self.hidden_size,
                    message_dim=self.message_size,
                    out_dim=self.hidden_size,
                    aggr=aggr,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.gnn_conv_out = CustomConv(
            in_dim=self.hidden_size,
            message_dim=self.message_size,
            out_dim=self.embedding_size,
            aggr=aggr,
        )

    def forward(self, x, edge_index, obj_ref):
        # Initialize embedding tensor
        x_e = torch.zeros(
            (x.shape[0], x.shape[1], self.embedding_size), device=x.device.type
        )
        for k in range(x.shape[1]):
            x_k = self.gnn_conv_in(x[:, k, :], edge_index, obj_ref)
            for h in range(len(self.gnn_conv_hidden)):
                # for some strange reason the hidden convs have to be send to the device separately
                x_k = self.gnn_conv_hidden[h](x_k, edge_index, obj_ref)
            x_k = self.gnn_conv_out(x_k, edge_index, obj_ref)
            x_e[:, k, :] = x_k
        return x_e  # x_e shape: (N, seq_length, embedding_size)


class ML_Linear_Embedding(CustomModule):
    def __init__(self, input_features, num_hidden_layers, hidden_size, embedding_size):
        super(ML_Linear_Embedding, self).__init__()

        """
        Module description: The Multy layer (ML) Linear_Embedding Module transforms a node feature vector (of 
        dimension input_features) into a node feature embedding (of dimension embedding_size) via multiple Linear 
        layers with one RELU activation function after the first linear Layer. The forward method takes takes the 
        graph node feature tensor and outputs an embedded node feature tensor. 
        """

        self.input_features = input_features
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

        self.linear_in = nn.Sequential(
            nn.Linear(
                in_features=self.input_features,
                out_features=self.hidden_size,
                bias=True,
            ),
            nn.ReLU(),
        )

        self.linear_hidden = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.hidden_size,
                    out_features=self.hidden_size,
                    bias=True,
                )
                for _ in range(self.num_hidden_layers)
            ]
        )

        self.linear_out = nn.Linear(
            in_features=self.hidden_size, out_features=self.embedding_size, bias=True
        )

        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x_e = self.linear_in(x)
        for h in range(len(self.linear_hidden)):
            x_e = self.linear_hidden[h](x_e)
        x_e = self.leaky_relu(self.linear_out(x_e))
        return x_e  # x_e shape: (N, seq_length, embedding_size)


class SL_Linear_Embedding(CustomModule):
    def __init__(self, input_features, embedding_size):
        super(SL_Linear_Embedding, self).__init__()

        """
        Module description: The Single layer (SL) Linear_Embedding Module transforms a node feature vector (of 
        dimension input_features) into a node feature embedding (of dimension embedding_size) via one Linear layer 
        followed by a RELU activation function. The forward method takes takes the graph node feature tensor and 
        outputs an embedded node feature tensor. 
        """

        self.input_features = input_features
        self.embedding_size = embedding_size

        self.embedding = nn.Linear(
            in_features=self.input_features, out_features=self.embedding_size, bias=True
        )
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x_e = self.leaky_relu(self.embedding(x))
        return x_e  # x_e shape: (N, seq_length, embedding_size)
