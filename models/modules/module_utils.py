import os
import sys

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_path)

import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from utils.geometry import coord_to_coord


class CustomModule(nn.Module):
    def __init__(self):
        super(CustomModule, self).__init__()
        self.frozen = False

    # At this point I have to emphasize, that the freeze and unfreeze function is not tested well. I thought,
    # these functions come in handy, especially for Link prediction between agents. However, they are not actively
    # used, mostly since the Link prediction is not actively used as well.
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
            param.requires_grad_(False)
            self.frozen = True
        return None

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
            param.requires_grad_(True)
            self.frozen = False
        return None


class CustomConv(MessagePassing):
    def __init__(
        self,
        in_dim,
        message_dim,
        out_dim,
        aggr="max",
        message_coord_transformation=False,
    ):
        self.aggr = aggr
        super(CustomConv, self).__init__(aggr=self.aggr)

        self.in_dim = in_dim
        self.message_dim = message_dim
        self.out_dim = out_dim
        self.message_coord_transformation = message_coord_transformation

        self.phy_j = Sequential(
            Linear(self.in_dim, self.message_dim, bias=True), ReLU()
        )
        self.phy_i = Sequential(
            Linear(self.in_dim, self.message_dim, bias=True), ReLU()
        )

        self.gamma = Linear(self.message_dim, self.out_dim, bias=True)

    def forward(self, x, edge_index, obj_ref):
        return self.propagate(edge_index, x=x, obj_ref=obj_ref.squeeze(1)[:, 6:9])

    def message(self, x_j, x_i, obj_ref_i, obj_ref_j):
        if self.message_coord_transformation:
            if x_j.shape[1] > 4:
                agent_v_inp = (x_j[:, 3], x_j[:, 4])
                agent_a_inp = (x_j[:, 5], x_j[:, 6])
            else:
                agent_v_inp = None
                agent_a_inp = None

            # Transform data of message omitting node to coordinate system of receiving node
            outp_ = coord_to_coord(
                new_coord=(obj_ref_i[:, 0], obj_ref_i[:, 1], obj_ref_i[:, 2]),
                old_coord=(obj_ref_j[:, 0], obj_ref_j[:, 1], obj_ref_j[:, 2]),
                agent_p=(x_j[:, 0], x_j[:, 1], x_j[:, 2]),
                agent_v=agent_v_inp,
                agent_a=agent_a_inp,
            )

            if x_j.shape[1] > 4:
                (
                    (x_j[:, 0], x_j[:, 1], x_j[:, 2]),
                    (x_j[:, 3], x_j[:, 4]),
                    (x_j[:, 5], x_j[:, 6]),
                ) = outp_
            else:
                (x_j[:, 0], x_j[:, 1], x_j[:, 2]) = outp_

        return self.phy_j(x_j)

    def update(self, aggr_out, x):
        x = self.phy_i(x) + aggr_out
        return self.gamma(x)
