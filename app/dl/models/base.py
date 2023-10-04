from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SelfAttentionBlockInfo:
    input_size: int
    hidden_size: int
    num_head: int
    dropout: float


class SelfAttentionBlock(nn.Module):
    def __init__(self, info: SelfAttentionBlockInfo):
        super(SelfAttentionBlock, self).__init__()

        self.info = info
        input_size = self.info.input_size
        hidden_size = self.info.hidden_size
        num_head = self.info.num_head
        dropout = self.info.dropout

        at_info = AttentionBlockInfo(
            query_size=input_size,
            key_size=input_size,
            value_size=input_size,
            hidden_size=hidden_size,
            num_head=num_head,
            dropout=dropout,
        )
        self.attention = AttentionBlock(at_info)

    def forward(self, x):
        output = self.attention(x, x, x)
        return output


@dataclass
class AttentionBlockInfo:
    query_size: int
    key_size: int
    value_size: int
    hidden_size: int
    num_head: int
    dropout: float


class AttentionBlock(nn.Module):
    def __init__(self, info: AttentionBlockInfo):
        super(AttentionBlock, self).__init__()

        self.info = info
        query_size = self.info.query_size
        key_size = self.info.key_size
        value_size = self.info.value_size
        hidden_size = self.info.hidden_size
        num_head = self.info.num_head
        dropout = self.info.dropout

        self.query_layer = nn.Linear(query_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.key_layer = nn.Linear(key_size, hidden_size)
        self.value_layer = nn.Linear(value_size, hidden_size)

        self.mha = nn.MultiheadAttention(
            hidden_size, num_head, batch_first=True)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value):
        seq_len = query.size(1)

        q = self.query_layer(query)
        q = self.dropout(q)
        k = self.key_layer(key)
        k = self.dropout(k)
        v = self.value_layer(value)
        v = self.dropout(v)

        z, _ = self.mha(q, k, v)
        x = v + z

        x = torch.stack([self.norm(x[:, i, :]) for i in range(seq_len)], dim=1)
        z = self.fc2(x)
        z = self.dropout(z)
        z = self.relu(z)
        z = self.fc3(z)
        z = self.dropout(z)
        x = x + z

        output = torch.stack([self.norm(x[:, i, :])
                             for i in range(seq_len)], dim=1)
        return output


@dataclass
class LSTMBlockInfo:
    input_size: int
    hidden_size: int
    dropout: float


class LSTMBlock(nn.Module):
    def __init__(self, info: LSTMBlockInfo):
        super(LSTMBlock, self).__init__()

        self.info = info
        input_size = self.info.input_size
        hidden_size = self.info.hidden_size
        droput = self.info.dropout

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=droput)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(droput)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.fc1(x)
        z, _ = self.lstm(x)
        z = self.fc2(z)
        z = self.dropout(z)

        x = torch.stack([self.norm(x[:, i, :]) for i in range(seq_len)], dim=1)
        z = self.fc3(z)
        z = self.dropout(z)
        z = self.relu(z)
        x = x + z

        output = torch.stack([self.norm(x[:, i, :])
                             for i in range(seq_len)], dim=1)
        return output


class PositionEncoder(nn.Module):
    def __init__(self, num_classes, output_dim):
        super(PositionEncoder, self).__init__()
        self.num_classes = num_classes
        self.output_dim = output_dim

        self.model = nn.Embedding(
            num_classes + 1, output_dim, padding_idx=0)  # use zero padding

    def forward(self, x):
        x = self.model(x)
        return x
