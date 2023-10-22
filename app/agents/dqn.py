from dataclasses import dataclass
from typing import List, Union

import torch

from app.utils import utils
from app.types import State, Action
from app.agents.agent import Agent
from app.dl.models.dqn import DQNValue, DQNValueInfo, DQNAdvantage, DQNAdvantageInfo
from app.dl.models.core import StateEncoder, StateEncoderInfo
from app.constants import *


@dataclass
class DQNAgentInfo:
    encoder_info: StateEncoderInfo
    vnf_s_advantage_info: DQNAdvantageInfo
    vnf_p_advantage_info: DQNAdvantageInfo
    vnf_s_value_info: DQNValueInfo
    vnf_p_value_info: DQNValueInfo


class DQNAgent(Agent):
    name = "DRL (DQN)"

    def __init__(self, info: DQNAgentInfo):
        self.info = info

        self.encoder = StateEncoder(info.encoder_info)
        self.vnf_s_advantage = DQNAdvantage(info.vnf_s_advantage_info)
        self.vnf_p_advantage = DQNAdvantage(info.vnf_p_advantage_info)
        self.vnf_s_value = DQNValue(info.vnf_s_value_info)
        self.vnf_p_value = DQNValue(info.vnf_p_value_info)

    def inference(self, input: Union[List[List[State]], List[State], State]) -> Union[Action, List[Action]]:
        if isinstance(input, State):
            input = [[input]]
        elif isinstance(input[0], State):
            input = [[state] for state in input]
        
        
        self.encoder.eval()
        self.vnf_s_advantage.eval()
        self.vnf_p_advantage.eval()

        rack_x, srv_x, sfc_x, vnf_x, core_x = self.encoder(
            input)
        vnf_s_advantage = self.vnf_s_advantage(
            core_x.unsqueeze(1).repeat(1, vnf_x.shape[1], 1),
            vnf_x,
            vnf_x.clone(),
        )
        vnf_s_value = self.vnf_s_value(core_x)
        vnf_s_q = vnf_s_value + vnf_s_advantage - vnf_s_advantage.mean(dim=1, keepdim=True)

        vnf_p_advantage = self.vnf_p_advantage(
            vnf_s_q.unsqueeze(1).repeat(1, srv_x.shape[1], 1),
            srv_x,
            srv_x.clone(),
        )
        vnf_p_value = self.vnf_p_value(vnf_s_q)
        vnf_p_q = vnf_p_value + vnf_p_advantage - vnf_p_advantage.mean(dim=1, keepdim=True)
        
        action_mask = utils.get_possible_action_mask(input).to(TORCH_DEVICE)
        vnf_s_mask = action_mask.sum(dim=2) == 0
        masked_vnf_s_q = vnf_s_q.masked_fill(vnf_s_mask, -1e+9)
        vnf_idxs = masked_vnf_s_q.argmax(dim=1)
        vnf_p_mask = action_mask[torch.arange(vnf_idxs.shape[0]), vnf_idxs, :] == 0
        masked_vnf_p_q = vnf_p_q.masked_fill(vnf_p_mask, -1e+9)
        srv_idxs = masked_vnf_p_q.argmax(dim=1)

        actions = [
            Action(
                vnfId=input[idx][-1].vnfList[vnf_idx].id, 
                srvId=input[idx][-1].get_srvList()[srv_idx].id
            ) for idx, (vnf_idx, srv_idx) in enumerate(zip(vnf_idxs, srv_idxs))
        ]

        if len(actions) == 1:
            return actions[0]
        
        return actions
