import os
import sys

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

import torch
from models.modules.Embedding_Modules import ML_Linear_Embedding
from models.modules.LSTM_Encoder_Modules import LSTM_Encoder
from models.modules.LSTM_Decoder_Modules import Indy_Decoder
from models.modules.SCIMG_Encoder import SCIMG_Encoder
from models.model_utils import CustomModel


class LSTM_Model(CustomModel):
    def __init__(
        self,
        input_features=8,
        linear_hidden_size=50,
        linear_hidden_layers=1,
        embedding_size=50,
        lstm_hidden_size=50,
        lstm_num_layers=2,
        output_features=2,
        sc_img=False,
    ):
        super(LSTM_Model, self).__init__()

        """
        Model description:
        The Vanilla_LSTM Model uses a LSTM_Encoder Module to encode the past data information in LSTM hidden and cell
        states which are passed to a LSTM_Decoder Module that predicts future trajectories.
        """

        self.tag = "linear_lstm"

        self.input_features = input_features
        self.linear_hidden_size = linear_hidden_size
        self.linear_hidden_layers = linear_hidden_layers
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.output_features = output_features

        self.Encoder_Embedding = ML_Linear_Embedding(
            input_features=self.input_features,
            num_hidden_layers=self.linear_hidden_layers,
            hidden_size=self.linear_hidden_size,
            embedding_size=self.embedding_size,
        )

        self.Encoder = LSTM_Encoder(
            embedding_size=self.embedding_size,
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

        # Linear Embedding
        x_e = self.Encoder_Embedding(x)

        # LSTM Encoder
        hidden, cell = self.Encoder(x_e)

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

    model = LSTM_Model(
        input_features=net_config["input_features"],
        linear_hidden_size=net_config["linear_hidden_size"],
        linear_hidden_layers=net_config["linear_hidden_layers"],
        embedding_size=net_config["input_embedding_size"],
        lstm_hidden_size=net_config["lstm_hidden_size"],
        lstm_num_layers=net_config["lstm_num_layers"],
        output_features=net_config["output_features"],
        sc_img=cfg_train["sc_img"],
    )

    model.to(cfg_train["device"])

    optimizer = optim.Adam(model.parameters(), lr=cfg_train["base_lr"])

    iter_loss = -1.0
    # send batch to device
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
