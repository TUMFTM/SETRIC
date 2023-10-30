import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from models.model_utils import CustomModel
from models.modules.LSTM_Decoder_Modules import Indy_Decoder


class LSTMindy_Model(CustomModel):
    def __init__(self, output_features=2):
        super(LSTMindy_Model, self).__init__()

        """
        Model description: 
        The LSTMIndy-Model uses a LSTM_Encoder Module to encode the past data information in LSTM hidden and cell 
        states which are passed to a LSTM_Decoder Indy_Decoder that predicts future trajectories.
        """

        self.tag = "lstm_indy"

        self.encoder_size = 64
        self.dyn_embedding_size = 32
        self.input_embedding_size = 32

        self.output_features = output_features

        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)

        self.enc_lstm_hist = torch.nn.LSTM(
            self.input_embedding_size, self.encoder_size, 1, batch_first=True
        )

        # Encoder hidden state embedder:
        self.dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)

        # Decoder LSTM
        self.decoder = Indy_Decoder()

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, batch):
        hist = batch.x[:, :, :2]

        _, (hist_enc, _) = self.enc_lstm_hist(self.leaky_relu(self.ip_emb(hist)))

        # torch.Size([13, 16, 2])
        # torch.Size([1, 13, 64])
        enc = self.leaky_relu(self.dyn_emb(hist_enc))
        # torch.Size([1, 13, 32])

        return self.decoder(enc)


if __name__ == "__main__":
    # Debugging of the model
    import torch.optim as optim
    from utils.processing import get_debug_data

    batch, cfg_train, net_config = get_debug_data(data="cr")

    model = LSTMindy_Model()

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
