from typing import List
from dataclasses import dataclass
from app.dl.models.base import AttentionBlock, AttentionBlockInfo

import torch.nn as nn


@dataclass
class DQNValueInfo:
    query_size: int
    key_size: int
    value_size: int
    hidden_sizes: List[int]
    num_heads: List[int]


class DQNValue(nn.Module):
    def __init__(self, info: DQNValueInfo):
        self.info = info

        self.models = nn.ModuleList()
        self.models.append(AttentionBlock(AttentionBlockInfo(
            info.query_size,
            info.key_size,
            info.value_size,
            info.hidden_sizes[0],
            info.num_heads[0],
            info.dropout,
        )))

        for i in range(1, len(info.hidden_sizes)):
            self.models.append(AttentionBlock(AttentionBlockInfo(
                info.query_size,
                info.hidden_sizes[i-1],
                info.hidden_sizes[i-1],
                info.hidden_sizes[i],
                info.num_heads[i],
                info.dropout,
            )))
        self.output_layer = nn.Linear(info.hidden_sizes[-1], 1)

    def forward(self, query, key, value):
        for model in self.models:
            key = model(query, key, value)
            value = key.clone()
        output = self.output_layer(key)
        output = output.squeeze(2)
        return output
