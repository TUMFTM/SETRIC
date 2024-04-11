import os
import sys

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

import torch
from models.modules.Embedding_Modules import GNN_Embedding_CustomConv
from models.modules.LSTM_Encoder_Modules import LSTM_Encoder
from models.modules.LSTM_Decoder_Modules import Indy_Decoder
from models.modules.Link_Pred_Modules import Distance_Edge_Module
from models.modules.SCIMG_Encoder import SCIMG_Encoder
from models.model_utils import CustomModel


class D_GNN_LSTM_Model(CustomModel):
    def __init__(
        self,
        input_features=8,
        gnn_d=10,
        gnn_num_hidden_layers=1,
        gnn_message_size=50,
        gnn_hidden_size=50,
        gnn_embedding_size=50,
        gnn_aggr="add",
        lstm_hidden_size=50,
        lstm_num_layers=2,
        output_features=2,
        sc_img=False,
    ):
        super(D_GNN_LSTM_Model, self).__init__()

        """
        Model description: The D_GNN_LSTM Model uses a LSTM_Encoder Module to encode the past data information in
        LSTM hidden and cell states which are passed to a LSTM_Decoder Module that predicts future trajectories. In
        addition to the GNN embedding of the GNN_LSTM Model, Edges are only considered between agents that are within
        a certain proximity.
        """

        self.tag = "d_gnn_lstm"

        self.input_features = input_features
        self.gnn_d = gnn_d
        self.gnn_num_hidden_layers = gnn_num_hidden_layers
        self.gnn_message_size = gnn_message_size
        self.gnn_hidden_size = gnn_hidden_size
        self.gnn_embedding_size = gnn_embedding_size
        self.gnn_aggr = gnn_aggr
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.output_features = output_features

        self.Link_Pred = Distance_Edge_Module(distance=self.gnn_d)

        self.Encoder_Embedding = GNN_Embedding_CustomConv(
            input_features=self.input_features,
            message_size=self.gnn_message_size,
            aggr=self.gnn_aggr,
            embedding_size=self.gnn_embedding_size,
            num_hidden_layers=self.gnn_num_hidden_layers,
            hidden_size=self.gnn_hidden_size,
        )

        self.Encoder = LSTM_Encoder(
            embedding_size=self.gnn_embedding_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
        )

        # Encoder hidden state embedder
        self.dyn_emb = torch.nn.Linear(self.lstm_hidden_size, 32)
        self.Decoder = Indy_Decoder(dyn_embedding_size=32 + 32 * int(sc_img))

        self.sc_img = sc_img

        if self.sc_img:
            self.scimg_encoder = SCIMG_Encoder()

        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, batch):
        x = batch.x  # x shape: (N, seq_length, input_features)

        # Link Prediction
        if not hasattr(
            batch, "ptr"
        ):  # In case no dataloader with batches is used, ptr might not be available
            batch.ptr = torch.tensor([0, batch.x.shape[0]])
        edge_index_pred = self.Link_Pred(
            x_time_series=x, ptr=batch.ptr, obj_ref=batch.obj_ref
        )

        # GNN Embedding
        x_e = self.Encoder_Embedding(x, edge_index_pred, batch.obj_ref)

        # LSTM Encoder
        hidden, cell = self.Encoder(x_e)

        # Scene Encoding
        if self.sc_img:
            sc_img_enc = self.scimg_encoder(batch.sc_img)
        else:
            sc_img_enc = None

        # LSTM Decoder
        enc = self.leaky_relu(
            self.dyn_emb(hidden.view(hidden.shape[1], hidden.shape[2]))
        )
        if sc_img_enc is not None:
            enc = torch.cat((enc, sc_img_enc), 1)
        pred = self.Decoder(enc)

        return pred


if __name__ == "__main__":
    # Debugging of the model
    import torch.optim as optim
    from utils.processing import get_debug_data

    batch, cfg_train, net_config = get_debug_data(data="cr")

    model = D_GNN_LSTM_Model(
        input_features=net_config["input_features"],
        gnn_d=net_config["gnn_distance"],
        gnn_num_hidden_layers=net_config["gnn_num_hidden_layers"],
        gnn_message_size=net_config["gnn_message_size"],
        gnn_hidden_size=net_config["gnn_hidden_size"],
        gnn_embedding_size=net_config["gnn_embedding_size"],
        gnn_aggr=net_config["gnn_aggr"],
        lstm_hidden_size=net_config["lstm_hidden_size"],
        lstm_num_layers=net_config["lstm_num_layers"],
        output_features=net_config["output_features"],
        sc_img=cfg_train["sc_img"],
    )

    model.to(cfg_train["device"])
    optimizer = optim.Adam(model.parameters(), lr=cfg_train["base_lr"])

    iter_loss = -1.0
    batch = batch.to(cfg_train["device"])
    for j in range(cfg_train["epochs"]):
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Compute prediction
        pred = model(batch)
        # Compute loss
        train_loss_batch = model.loss(pred, batch.y[:, :, :2])
        iter_loss = train_loss_batch / batch.x.shape[0]

        # backpropagation
        train_loss_batch.backward()

        # Gradient Clipping
        if cfg_train["clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        # optimization step
        optimizer.step()

        print("Iter: {:03d}, Loss: {:.02f}".format(j + 1, iter_loss), end="\r")

    print("Iter: {:03d}, Loss: {:.02f}".format(j + 1, iter_loss))
    print(cfg_train)
