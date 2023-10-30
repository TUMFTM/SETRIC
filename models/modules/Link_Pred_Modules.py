import os
import sys

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_path)

import torch
import torch.nn as nn
from utils.processing import get_edge_index
from utils.geometry import coord_to_coord
from models.modules.module_utils import CustomModule
from models.modules.Embedding_Modules import GNN_Embedding_CustomConv


class GNN_Link_Pred_Module(CustomModule):
    def __init__(
        self,
        input_features,
        embedding_size,
        message_size,
        aggr,
        num_hidden_layers,
        hidden_size,
    ):
        super(GNN_Link_Pred_Module, self).__init__()

        """
        Module description: 
        The GNN_Link_Pred_Module takes a graph data object and performs link prediction between nodes. For this purpose,
        the node features are mapped into a high dimensional embedding space via multiple graph convolution layers. 
        Afterwards, the decode_all method determines edges between those nodes with close proximity in the embedding 
        space.
        """

        self.input_features = input_features
        self.embedding_size = embedding_size
        self.message_size = message_size
        self.aggr = aggr
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

        self.Lstm_Time_series_Encoder = nn.LSTM(
            input_size=self.input_features,
            hidden_size=self.input_features,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        self.Lstm_Single_Time_Decoder = nn.LSTM(
            input_size=self.input_features,
            hidden_size=self.input_features,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        self.Encoder = GNN_Embedding_CustomConv(
            input_features=self.input_features,
            message_size=self.message_size,
            aggr=self.aggr,
            embedding_size=self.embedding_size,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
        )

    def forward(self, x_time_series, ptr, obj_ref):
        _, (hidden, cell) = self.Lstm_Time_series_Encoder(x_time_series)
        x, (_, __) = self.Lstm_Single_Time_Decoder(
            x_time_series[:, -1, :].unsqueeze(1), (hidden, cell)
        )
        x = x.detach()
        edge_index_list = []
        # split batch of graphs into single graphs
        for k in range(len(ptr) - 1):
            x_subgraph = x[ptr[k] : ptr[k + 1], :, :]
            obj_ref_subgraph = obj_ref[ptr[k] : ptr[k + 1], :, :]
            # Only predict edges if more than one agent is present
            if x_subgraph.shape[0] > 1:
                edge_index_subgraph = (
                    get_edge_index(x_subgraph.shape[0]).t().contiguous()
                )
                z = self.Encoder(
                    x_subgraph, edge_index_subgraph.to(x.device.type), obj_ref_subgraph
                ).squeeze(1)
                edge_index_subgraph_pred = self.decode_all(z)
                # The edge_index list starts with 0 for every subgraph, thus the number of the node with subgraph
                # index 0 within the entire batch graph has to be added to every index
                edge_index_list.append(edge_index_subgraph_pred + ptr[k])
            else:
                edge_index_list.append(
                    torch.tensor(
                        [[], []], dtype=torch.long, device=x_time_series.device.type
                    )
                )
        return torch.cat(edge_index_list, dim=1)  # Concatenate edge_index_list

    @staticmethod
    def decode_all(z):
        prob_adj = z @ z.t()  # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t()  # get predicted edge_list


class Distance_Edge_Module(CustomModule):
    def __init__(self, distance):
        super(Distance_Edge_Module, self).__init__()

        """
        Module description: The Distance_Edge_Module takes a graph data object and determines the euclidean 
        distance between agents. Edges are only formed between agents that are within a certain distance at the current 
        time step.
        """

        self.distance = distance

    def forward(self, x_time_series, ptr, obj_ref):
        # Transform to UTM
        x = torch.zeros_like(obj_ref[:, 0, :2])
        (x[:, 0], x[:, 1], _) = coord_to_coord(
            new_coord=(0, 0, 0),
            old_coord=(obj_ref[:, 0, 6], obj_ref[:, 0, 7], obj_ref[:, 0, 8]),
            agent_p=(
                torch.tensor(0).float(),
                torch.tensor(0).float(),
                torch.tensor(0).float(),
            ),
        )
        edge_index_list = []
        # split batch of graphs into single graphs
        for k in range(len(ptr) - 1):
            x_subgraph = x[ptr[k] : ptr[k + 1], :]
            # Only predict edges if more than one agent is present
            if x_subgraph.shape[0] > 1:
                edge_index_subgraph_pred = (
                    (torch.cdist(x_subgraph, x_subgraph, p=2) < self.distance)
                    .nonzero(as_tuple=False)
                    .t()
                )
                # The edge_index list starts with 0 for every subgraph, thus the number of the node with subgraph
                # index 0 within the entire batch graph has to be added to every index
                edge_index_list.append(edge_index_subgraph_pred + ptr[k])
            else:
                edge_index_list.append(
                    torch.tensor(
                        [[], []], dtype=torch.long, device=x_time_series.device.type
                    )
                )
        return torch.cat(edge_index_list, dim=1)  # Concatenate edge_index_list
