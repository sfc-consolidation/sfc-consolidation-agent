from dataclasses import dataclass
from typing import List, Union

from app.utils import utils
from app.types import State, Action
from app.agents.agent import Agent
from app.dl.models.dqn import DQNValue, DQNValueInfo
from app.dl.models.core import StateEncoder, StateEncoderInfo


@dataclass
class DQNAgentInfo:
    encoder_info: StateEncoderInfo
    vnf_s_value_info: DQNValueInfo
    vnf_p_value_info: DQNValueInfo


class DQNAgent(Agent):
    name = "DRL (DQN)"

    def __init__(self, info: DQNAgentInfo):
        self.info = info

        self.encoder = StateEncoder(info.encoder_info)
        self.vnf_s_value = DQNValue(info.vnf_s_value_info)
        self.vnf_p_value = DQNValue(info.vnf_p_value_info)

    def inference(self, input: Union[List[List[State]], List[State], State]) -> Action:
        action_mask = utils.get_possible_action_mask(input)
        
        self.encoder.eval()
        self.vnf_s_value.eval()
        self.vnf_p_value.eval()

        rack_x, srv_x, sfc_x, vnf_x, core_x = self.encoder(
            input)
        vnf_s_value = self.vnf_s_value(
            core_x.unsqueeze(1).repeat(1, vnf_x.shape[1], 1),
            vnf_x,
            vnf_x.clone(),
        )
        vnf_s_mask = action_mask.sum(dim=3) == 0
        vnf_s_value = vnf_s_value.masked_fill(vnf_s_mask, -1e9)
        vnf_id = int(vnf_s_value.argmax())
        vnf_p_value = self.vnf_p_value(
            vnf_s_value.unsqueeze(1).repeat(1, srv_x.shape[1], 1),
            srv_x,
            srv_x.clone(),
        )
        vnf_p_mask = action_mask[:, :, vnf_id, :] == 0
        vnf_p_value = vnf_p_value.masked_fill(vnf_p_mask, -1e9)
        vnf_p_action = vnf_p_value.argmax()
        srv_id = int(vnf_p_action)

        return Action(vnf_id, srv_id)
