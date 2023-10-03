from typing import List
from dataclasses import dataclass

import torch.nn as nn

from app.dl.models.base import AttentionBlock, AttentionBlockInfo


@dataclass
class PPOValueInfo:
    query_size: int
    key_size: int
    value_size: int
    hidden_sizes: List[int]
    num_heads: List[int]
    dropout: float
    seq_len: int


class PPOValue(nn.Module):
    def __init__(self, info: PPOValueInfo):
        super(PPOValue, self).__init__()

        self.info = info

        self.models = nn.ModuleList()
        self.models.append(
            AttentionBlock(
                AttentionBlockInfo(
                    info.query_size,
                    info.key_size,
                    info.value_size,
                    info.hidden_sizes[0],
                    info.num_heads[0],
                    info.dropout
                )
            )
        )
        for i in range(1, len(info.hidden_sizes)):
            self.models.append(AttentionBlock(AttentionBlockInfo(
                info.query_size,
                info.hidden_sizes[i-1],
                info.hidden_sizes[i-1],
                info.hidden_sizes[i],
                info.num_heads[i],
                info.dropout,
            )))
        self.conv1d = nn.Conv2d(self.info.seq_len, 1, 1)
        self.output_layer = nn.Linear(info.hidden_sizes[-1], 1)

    def forward(self, query, key, value):
        for model in self.models:
            key = model(query, key, value)
            value = key.clone()
        output = self.conv1d(key)
        output = self.output_layer(output)
        output = output.squeeze()
        return output


@dataclass
class PPOPolicyInfo:
    query_size: int
    key_size: int
    value_size: int
    hidden_sizes: List[int]
    num_heads: List[int]
    dropout: float


class PPOPolicy(nn.Module):
    def __init__(self, info: PPOPolicyInfo):
        super(PPOPolicy, self).__init__()
        self.info = info

        self.models = nn.ModuleList()
        self.models.append(
            AttentionBlock(
                AttentionBlockInfo(
                    info.query_size,
                    info.key_size,
                    info.value_size,
                    info.hidden_sizes[0],
                    info.num_heads[0],
                    info.dropout
                )
            )
        )
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
        output = output.squeeze()
        return output
