from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SelfAttentionBlockInfo:
    input_size: int
    hidden_size: int
    num_head: int
    dropout: float
    device: str = 'cpu'


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
            device=self.info.device,
        )
        self.attention = AttentionBlock(at_info)


    def forward(self, x):
        x = self._format(x)
        output = self.attention(x, x.clone(), x.clone())
        return output

    def _format(self, x):
        return x.to(self.info.device)

@dataclass
class AttentionBlockInfo:
    query_size: int
    key_size: int
    value_size: int
    hidden_size: int
    num_head: int
    dropout: float
    device: str = 'cpu'


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

        self.query_layer = nn.Linear(query_size, hidden_size).to(self.info.device)
        self.dropout = nn.Dropout(dropout).to(self.info.device)

        self.key_layer = nn.Linear(key_size, hidden_size).to(self.info.device)
        self.value_layer = nn.Linear(value_size, hidden_size).to(self.info.device)

        self.mha = nn.MultiheadAttention(
            hidden_size, num_head, batch_first=True).to(self.info.device)
        self.norm = nn.BatchNorm1d(hidden_size).to(self.info.device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(self.info.device)
        self.relu = nn.ReLU().to(self.info.device)
        self.fc3 = nn.Linear(hidden_size, hidden_size).to(self.info.device)

    def forward(self, query, key, value):
        query = self._format(query)
        key = self._format(key)
        value = self._format(value)
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
    
    def _format(self, input):
        return input.to(self.info.device)


@dataclass
class LSTMBlockInfo:
    input_size: int
    hidden_size: int
    dropout: float
    device: str = 'cpu'


class LSTMBlock(nn.Module):
    def __init__(self, info: LSTMBlockInfo):
        super(LSTMBlock, self).__init__()

        self.info = info
        input_size = self.info.input_size
        hidden_size = self.info.hidden_size
        droput = self.info.dropout

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=droput).to(self.info.device)
        self.norm = nn.BatchNorm1d(hidden_size).to(self.info.device)
        self.fc1 = nn.Linear(input_size, hidden_size).to(self.info.device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(self.info.device)
        self.relu = nn.ReLU().to(self.info.device)
        self.fc3 = nn.Linear(hidden_size, hidden_size).to(self.info.device)
        self.dropout = nn.Dropout(droput).to(self.info.device)

    def forward(self, x):
        x = self._format(x)
        seq_len = x.size(1)
        x = self.fc1(x)
        z, _ = self.lstm(x)
        z = self.fc2(z)
        z = self.dropout(z)
        x = x + z

        x = torch.stack([self.norm(x[:, i, :]) for i in range(seq_len)], dim=1) # TODO: Expected more than 1 value per channel when training, got input size torch.Size([1, 32])
        z = self.fc3(z)
        z = self.dropout(z)
        z = self.relu(z)
        x = x + z

        output = torch.stack([self.norm(x[:, i, :])
                             for i in range(seq_len)], dim=1)
        return output

    def _format(self, x):
        return x.to(self.info.device)

@dataclass
class PositionEncoderInfo:
    num_classes: int
    output_size: int
    device: str = 'cpu'


class PositionEncoder(nn.Module):
    def __init__(self, info: PositionEncoderInfo):
        super(PositionEncoder, self).__init__()
        self.info = info
        num_classes = info.num_classes
        output_size = info.output_size

        self.model = nn.Embedding(
            num_classes + 1, output_size, padding_idx=0).to(self.info.device)  # use zero padding
        

    def forward(self, x):
        x = self._format(x)
        x = self.model(x)
        return x
    
    def _format(self, x):
        return x.to(self.info.device)
