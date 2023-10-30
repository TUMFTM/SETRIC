import os
import sys

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_path)

import torch
import torch.nn as nn
import random
from models.modules.module_utils import CustomModule


class LSTM_Decoder(CustomModule):
    def __init__(
        self, Embedding, hidden_size, num_layers, output_features, embedding_size=None
    ):
        super(LSTM_Decoder, self).__init__()

        """
        Module description: 
        The LSTM_Decoder Module takes the hidden end cell states outputted by a LSTM_Encoder that holds the encoded 
        information of past data. The LSTM net predicts one future time instance holding the positional information of 
        all agents in the batch using the past position (x, y) as input. The prediction is reused as the input of the 
        next prediction.
        """

        if embedding_size is None:
            self.embedding_size = Embedding.embedding_size
        else:
            self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_features = output_features

        self.teacher_force_ration = 0

        self.Embedding = Embedding

        self.Lstm_Decoder = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )

        self.fc = nn.Linear(
            in_features=self.hidden_size, out_features=self.output_features
        )

    def forward(self, hidden, cell, batch_size, pred_len, y=None, sc_img_enc=None):
        if self.training and self.teacher_force_ration != 0 and y is None:
            raise ValueError(
                "Invalid y - If model is in training mode, y can't be None"
            )

        # The predicted relative x and y positions are used as inputs for the next prediction. For the first prediction,
        # the last x and y positions of the input data series have to be used, which are by definition the current
        # position and thus 0.
        # p shape: (N, output_features)
        p = torch.zeros(batch_size, self.output_features, device=hidden.device.type)
        outputs = torch.zeros(
            batch_size, pred_len, self.output_features, device=hidden.device.type
        )

        # Decoder
        for t in range(pred_len):
            enc = self.Embedding(p)
            if sc_img_enc is not None:
                enc = torch.cat((self.Embedding(p), sc_img_enc), 1)
            p_e = enc.unsqueeze(1)  # p_e shape: (N, 1, embedding_size)

            output, (hidden, cell) = self.Lstm_Decoder(p_e, (hidden, cell))
            # Transform feature size from hidden_size to output_features
            output = self.fc(output.squeeze(1))
            outputs[:, t, :] = output
            p = (
                y[:, t, :2]
                if random.random() < self.teacher_force_ration and self.training
                else output
            )

        return outputs


class Indy_Decoder(CustomModule):
    def __init__(
        self,
        decoder_size=128,
        dyn_embedding_size=32,
        output_features=2,
        output_length=50,
    ):
        """
        Module description:
        The Indy Decoder with LSTM decoding to generate future trajectories.
        """

        super(Indy_Decoder, self).__init__()
        self.decoder_size = decoder_size
        self.dyn_embedding_size = dyn_embedding_size

        self.output_features = output_features
        self.output_length = output_length

        # Decoder LSTM
        self.dec_lstm = torch.nn.LSTM(dyn_embedding_size, self.decoder_size)

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size, 5)

    def forward(self, enc):
        enc = enc.repeat(self.output_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)  # (batch_size, pred_len, decoder_size)
        fut_pred = self.op(h_dec)  # (batch_size, pred_len, out_size)
        fut_pred = outputActivation(fut_pred)
        return fut_pred[:, :, : self.output_features]


def outputActivation(x):
    """Custom activation for output layer (Graves, 2015)."""
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out
