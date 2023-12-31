from typing import Literal, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchrl.modules as trm

from app.dl.models.base import SelfAttentionBlock, SelfAttentionBlockInfo, LSTMBlock, LSTMBlockInfo

ENCODING_METHOD = Literal["FC", "LSTM", "SA"]
ENCODING_LAST_ACTIVATION = Literal["RELU", "SOFTMAX", "NONE"]


@dataclass
class EncoderInfo:
    input_size: int
    output_size: int
    hidden_sizes: List[int]
    last_activation: ENCODING_LAST_ACTIVATION = "NONE"
    batch_norm: bool = True
    method: ENCODING_METHOD = "FC"
    dropout: float = 0.1
    num_head: int = 1
    device: str = 'cpu'


class Encoder(nn.Module):
    # Basically, Encoder has two layers: input layer and output layer. If you want to add hidden layers, you can add hidden layer with num_hidden_layer variable.

    def __init__(self, info: EncoderInfo):
        """
        Initializer of Encoder

        Args:
            input_size (int): input size
            output_size (int): output size
            hidden_sizes (List[int]): hidden size list. Defaluts to [].
            last_activation (ENCODING_LAST_ACTIVATION, optional): last activation function. Defaults to "NONE".
            batch_norm (bool, optional): whether to use batch normalization. Defaults to True.
            method (ENCODING_METHOD, optional): encoding method. Defaults to "FC".
            dropout (float, optional): dropout rate. Defaults to 0.1.
            num_head (int, optional): number of heads. Defaults to 1.  <- only for SA
        """
        super(Encoder, self).__init__()
        self.info = info
        input_size = info.input_size
        output_size = info.output_size
        hidden_sizes = info.hidden_sizes
        last_activation = info.last_activation
        batch_norm = info.batch_norm
        method = info.method
        dropout = info.dropout
        num_head = info.num_head

        self.models = nn.ModuleList()
        if (method == "FC"):
            if (len(hidden_sizes) == 0):
                self.models.append(
                    trm.NoisyLinear(input_size, output_size).to(self.info.device))
                self.models.append(nn.Dropout(dropout).to(self.info.device))
            else:
                self.models.append(
                    trm.NoisyLinear(input_size, hidden_sizes[0]).to(self.info.device))
                self.models.append(nn.Dropout(dropout).to(self.info.device))
                self.models.append(nn.ReLU().to(self.info.device))
                if (batch_norm):
                    self.models.append(nn.BatchNorm1d(hidden_sizes[0]).to(self.info.device))
                for i in range(1, len(hidden_sizes)):
                    self.models.append(
                        trm.NoisyLinear(hidden_sizes[i-1], hidden_sizes[i]).to(self.info.device))
                    self.models.append(nn.Dropout(dropout).to(self.info.device))
                    self.models.append(nn.ReLU().to(self.info.device))
                    if (batch_norm):
                        self.models.append(
                            nn.BatchNorm1d(hidden_sizes[i]).to(self.info.device))
                self.models.append(
                    trm.NoisyLinear(hidden_sizes[-1], output_size).to(self.info.device))
        elif (method == "LSTM"):
            if (len(hidden_sizes) == 0):
                self.models.append(
                    LSTMBlock(LSTMBlockInfo(input_size, output_size, dropout, device=self.info.device)))
            else:
                self.models.append(
                    LSTMBlock(LSTMBlockInfo(input_size, hidden_sizes[0], dropout, device=self.info.device)))
                for i in range(1, len(hidden_sizes)):
                    self.models.append(
                        LSTMBlock(LSTMBlockInfo(hidden_sizes[i-1], hidden_sizes[i], dropout, device=self.info.device)))
                self.models.append(
                    LSTMBlock(LSTMBlockInfo(hidden_sizes[-1], output_size, dropout, device=self.info.device)))
        elif (method == "SA"):
            if (len(hidden_sizes) == 0):
                self.models.append(SelfAttentionBlock(SelfAttentionBlockInfo(
                    input_size, hidden_sizes, num_head, dropout, device=self.info.device)))
            else:
                self.models.append(SelfAttentionBlock(SelfAttentionBlockInfo(
                    input_size, hidden_sizes[0], num_head, dropout, device=self.info.device)))
                for i in range(1, len(hidden_sizes)):
                    self.models.append(SelfAttentionBlock(SelfAttentionBlockInfo(
                        hidden_sizes[i-1], hidden_sizes[i], num_head, dropout, device=self.info.device)))
                self.models.append(SelfAttentionBlock(SelfAttentionBlockInfo(
                    hidden_sizes[-1], output_size, num_head, dropout, device=self.info.device)))
        else:
            raise ValueError("Invalid method")

        if last_activation == "RELU":
            self.models.append(nn.ReLU().to(self.info.device))
        elif last_activation == "SOFTMAX":
            self.models.append(nn.Softmax().to(self.info.device))


    def validate(self, x: torch.Tensor) -> None:
        """
        Validate input size
        if FC: input = (batch_size, input_channel, input_size)
        if LSTM: input = (batch_size, seq_len, input_channel, input_size)
        if SA: input = (batch_size, seq_len, input_channel, input_size)
        Args:
            x (torch.Tensor): input tensor

        Raises:
            ValueError: Invalid input size
        """
        if (self.info.method == "FC"):
            if (x.shape[1] != self.info.input_size):
                raise ValueError("Invalid input size")
        elif (self.info.method == "LSTM"):
            if (x.shape[2] != self.info.input_size):
                raise ValueError("Invalid input size")
        elif (self.info.method == "SA"):
            if (x.shape[2] != self.info.input_size):
                raise ValueError("Invalid input size")

    def _format(self, x):
        return x.to(self.info.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate(x)
        x = self._format(x)
        for model in self.models:
            x = model(x)
        return x

    def save(self, path: str) -> None:
        torch.save(self.models.state_dict(), path)

    def load(self, path: str) -> None:
        self.models.load_state_dict(torch.load(path))
