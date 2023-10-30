import torch.nn as nn
from models.modules.module_utils import CustomModule


class LSTM_Encoder(CustomModule):
    def __init__(self, embedding_size, hidden_size, num_layers):
        super(LSTM_Encoder, self).__init__()

        """
        Module description: 
        The LSTM_Encoder Module encodes the information of past data in the hidden and cell states of the LSTM cells.
        """
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.Lstm_Encoder = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )

    def forward(self, x_e):
        _, (hidden, cell) = self.Lstm_Encoder(x_e)
        return hidden, cell
