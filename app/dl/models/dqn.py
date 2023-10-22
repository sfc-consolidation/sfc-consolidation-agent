from typing import List
from dataclasses import dataclass
from app.dl.models.base import AttentionBlock, AttentionBlockInfo

import torch.nn as nn
import torchrl.modules as trm

@dataclass
class DQNValueInfo:
    input_size: int
    hidden_sizes: List[int]
    dropout: float
    device: str = 'cpu'

class DQNValue(nn.Module):
    def __init__(self, info: DQNValueInfo):
        super(DQNValue, self).__init__()
        self.info = info

        self.models = nn.ModuleList()
        self.models.append(trm.NoisyLinear(info.input_size, info.hidden_sizes[0]).to(self.info.device))
        self.models.append(nn.Dropout(info.dropout))
        self.models.append(nn.ReLU())

        for i in range(1, len(info.hidden_sizes)):
            self.models.append(trm.NoisyLinear(info.hidden_sizes[i-1], info.hidden_sizes[i]).to(self.info.device))
            self.models.append(nn.Dropout(info.dropout))
            self.models.append(nn.ReLU())
        self.output_layer = trm.NoisyLinear(info.hidden_sizes[-1], 1).to(self.info.device)


    def forward(self, input):
        x = self._format(input)
        for model in self.models:
            x = model(x)
        output = self.output_layer(x)
        return output

    def _format(self, input):
        return input.to(self.info.device)

@dataclass
class DQNAdvantageInfo:
    query_size: int
    key_size: int
    value_size: int
    hidden_sizes: List[int]
    num_heads: List[int]
    dropout: float
    device: str = 'cpu'

class DQNAdvantage(nn.Module):
    def __init__(self, info: DQNAdvantageInfo):
        super(DQNAdvantage, self).__init__()
        self.info = info

        self.models = nn.ModuleList()
        self.models.append(AttentionBlock(AttentionBlockInfo(
            info.query_size,
            info.key_size,
            info.value_size,
            info.hidden_sizes[0],
            info.num_heads[0],
            info.dropout,
            device=self.info.device,
        )))

        for i in range(1, len(info.hidden_sizes)):
            self.models.append(AttentionBlock(AttentionBlockInfo(
                info.query_size,
                info.hidden_sizes[i-1],
                info.hidden_sizes[i-1],
                info.hidden_sizes[i],
                info.num_heads[i],
                info.dropout,
                device=info.device,
            )))
        self.output_layer = trm.NoisyLinear(info.hidden_sizes[-1], 1).to(self.info.device)


    def forward(self, query, key, value):
        query = self._format(query)
        key = self._format(key)
        value = self._format(value)

        for model in self.models:
            key = model(query, key, value)
            value = key.clone()
        output = self.output_layer(key)
        output = output.squeeze(2)
        return output

    def _format(self, input):
        return input.to(self.info.device)
