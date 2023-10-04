from typing import Tuple, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn

from app.dl.models.base import PositionEncoder
from app.dl.models.encoder import Encoder, EncoderInfo
from app.types import State


@dataclass
class StateEncoderInfo:
    max_rack_num: int
    rack_id_dim: int
    max_srv_num: int
    srv_id_dim: int
    srv_encoder_info: EncoderInfo
    max_vnf_num: int
    vnf_id_dim: int
    vnf_encoder_info: EncoderInfo
    max_sfc_num: int
    sfc_id_dim: int
    sfc_encoder_info: EncoderInfo
    core_encoder_info: EncoderInfo


class StateEncoder(nn.Module):
    def __init__(self, info: StateEncoderInfo):
        super(StateEncoder, self).__init__()
        self.info = info

        max_rack_num = info.max_rack_num
        rack_id_dim = info.rack_id_dim

        max_srv_num = info.max_srv_num
        srv_id_dim = info.srv_id_dim
        srv_encoder_info = info.srv_encoder_info
        max_vnf_num = info.max_vnf_num
        vnf_id_dim = info.vnf_id_dim
        vnf_encoder_info = info.vnf_encoder_info
        max_sfc_num = info.max_sfc_num
        sfc_id_dim = info.sfc_id_dim
        sfc_encoder_info = info.sfc_encoder_info
        core_encoder_info = info.core_encoder_info

        self.rack_pos_encoder = PositionEncoder(max_rack_num, rack_id_dim)
        self.srv_pos_encoder = PositionEncoder(max_srv_num, srv_id_dim)
        self.vnf_pos_encoder = PositionEncoder(max_vnf_num, vnf_id_dim)
        self.sfc_pos_encoder = PositionEncoder(max_sfc_num, sfc_id_dim)

        self.srv_encoder = Encoder(srv_encoder_info)
        self.vnf_encoder = Encoder(vnf_encoder_info)
        self.sfc_encoder = Encoder(sfc_encoder_info)

        self.core_encoder = Encoder(core_encoder_info)

    def forward(self, input: Union[List[List[State]], List[State], State]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        rack_x, srv_x, sfc_x, vnf_x = self._format(
            input)
        batch_size = rack_x.shape[0]
        seq_len = rack_x.shape[1]

        # Rack Info only has Rack ID
        rack_x = self.rack_pos_encoder(rack_x.to(torch.int32)).squeeze(3)

        srv_idxs = srv_x[:, :, :, 0]
        srv_x = torch.concat(
            [srv_x[:, :, :, 1:], self.srv_pos_encoder(srv_idxs.to(torch.int32))], dim=3)
        srv_x = srv_x.view(batch_size * seq_len,
                           srv_x.shape[2], srv_x.shape[3])
        srv_x = self.srv_encoder(srv_x)
        srv_x = srv_x.view(batch_size, seq_len, srv_x.shape[1], srv_x.shape[2])

        sfc_idxs = sfc_x[:, :, :, 0]
        sfc_x = torch.concat(
            [sfc_x[:, :, :, 1:], self.sfc_pos_encoder(sfc_idxs.to(torch.int32))], dim=3)
        sfc_x = sfc_x.view(batch_size * seq_len,
                           sfc_x.shape[2], sfc_x.shape[3])
        sfc_x = self.sfc_encoder(sfc_x)
        sfc_x = sfc_x.view(batch_size, seq_len, sfc_x.shape[1], sfc_x.shape[2])

        vnf_idxs = vnf_x[:, :, :, 0]
        vnf_srv_idxs = vnf_x[:, :, :, 1]
        vnf_sfc_idxs = vnf_x[:, :, :, 2]
        vnf_order_in_sfc = vnf_x[:, :, :, 3]
        vnf_x = torch.concat([
            vnf_x[:, :, :, 4:],
            self.vnf_pos_encoder(vnf_idxs.to(torch.int32)),
            self.srv_pos_encoder(vnf_srv_idxs.to(torch.int32)),
            self.sfc_pos_encoder(vnf_sfc_idxs.to(torch.int32)),
            self.sfc_pos_encoder(vnf_order_in_sfc.to(torch.int32))
        ], dim=3)
        vnf_x = vnf_x.view(batch_size * seq_len,
                           vnf_x.shape[2], vnf_x.shape[3])
        vnf_x = self.vnf_encoder(vnf_x)
        vnf_x = vnf_x.view(batch_size, seq_len, vnf_x.shape[1], vnf_x.shape[2])

        core_x = torch.concat([
            rack_x.view(batch_size, seq_len, -1),
            srv_x.view(batch_size, seq_len, -1),
            sfc_x.view(batch_size, seq_len, -1),
            vnf_x.view(batch_size, seq_len, -1),
        ], dim=2)

        core_x = self.core_encoder(core_x)

        return rack_x[:, -1], srv_x[:, -1], sfc_x[:, -1], vnf_x[:, -1], core_x[:, -1]

    def _format(self, input: Union[List[List[State]], List[State], State]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Format input data to tensor

        Args:
            input (Union[List[List[State]], List[State], State]): input data

        Returns:
            torch.Tensor: rack_x    (BATCH_LEN, SEQ_LEN, MAX_RACK_NUM, 1)
            torch.Tensor: srv_x     (BATCH_LEN, SEQ_LEN, MAX_SRV_NUM,  4)
            torch.Tensor: sfc_x     (BATCH_LEN, SEQ_LEN, MAX_SFC_NUM,  2)
            torch.Tensor: vnf_x     (BATCH_LEN, SEQ_LEN, MAX_VNF_NUM,  7)
        """
        if isinstance(input, State):
            input = [input]
        if not isinstance(input[0], list):
            input = [input]
        rack_x = torch.zeros(len(input), len(
            input[0]), self.info.max_rack_num, 1)
        srv_x = torch.zeros(len(input), len(
            input[0]), self.info.max_srv_num, 4)
        sfc_x = torch.zeros(len(input), len(
            input[0]), self.info.max_sfc_num, 2)
        vnf_x = torch.zeros(len(input), len(
            input[0]), self.info.max_vnf_num, 7)
        for batch_idx in range(len(input)):
            for seq_idx in range(len(input[batch_idx])):
                state = input[batch_idx][seq_idx]
                state_tensors = state.to_tensor()
                rack_x[batch_idx, seq_idx, :len(
                    state_tensors[0]), :] = state_tensors[0]
                srv_x[batch_idx, seq_idx, :len(
                    state_tensors[1]), :] = state_tensors[1]
                sfc_x[batch_idx, seq_idx, :len(
                    state_tensors[2]), :] = state_tensors[2]
                vnf_x[batch_idx, seq_idx, :len(
                    state_tensors[3]), :] = state_tensors[3]
        return rack_x, srv_x, sfc_x, vnf_x


@dataclass
class StopperInfo:
    input_size: int
    hidden_sizes: List[int]
    dropout: float


class Stopper(nn.Module):
    def __init__(self, info: StopperInfo):
        super(Stopper, self).__init__()
        self.info = info
        input_size = info.input_size
        hidden_sizes = info.hidden_sizes
        dropout = info.dropout

        self.models = nn.ModuleList()
        self.models.append(
            nn.Linear(input_size, hidden_sizes[0]))
        self.models.append(nn.Dropout(dropout))
        self.models.append(nn.ReLU())
        for i in range(1, len(hidden_sizes)):
            self.models.append(
                nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.models.append(nn.Dropout(dropout))
            self.models.append(nn.ReLU())
        self.models.append(nn.Linear(hidden_sizes[-1], 1))
        self.models.append(nn.Sigmoid())

    def forward(self, input):
        for model in self.models:
            input = model(input)
        return input.squeeze()
