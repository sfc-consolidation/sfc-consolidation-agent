from dataclasses import dataclass
from typing import List, Union

from app.types import State, Action
from app.agents.agent import Agent
from app.dl.models.dqn import DQNValue, DQNValueInfo
from app.dl.models.core import StateEncoder, StateEncoderInfo, Stopper, StopperInfo


@dataclass
class DQNAgentInfo:
    encoder_info: StateEncoderInfo
    stopper_info: StopperInfo
    vnf_s_value_info: DQNValueInfo
    vnf_p_value_info: DQNValueInfo


class DQNAgent(Agent):
    name = "DRL (DQN)"

    def __init__(self, info: DQNAgentInfo):
        self.info = info

        self.encoder = StateEncoder(info.encoder_info)
        self.stopper = Stopper(info.stopper_info)
        self.vnf_s_value = DQNValue(info.vnf_s_value_info)
        self.vnf_p_value = DQNValue(info.vnf_p_value_info)

    def inference(self, input: Union[List[List[State]], List[State], State]) -> Action:
        rack_x, srv_x, sfc_x, vnf_x, core_x = self.encoder(input)
        stop = self.stopper(core_x)
        if (stop):
            return None
        vnf_s_value = self.vnf_s_value(core_x, vnf_x, vnf_x.clone())
        vnf_id = int(vnf_s_value.argmax())
        vnf_p_value = self.vnf_p_value(vnf_s_value, srv_x, srv_x.clone())
        vnf_p_action = vnf_p_value.argmax()
        srv_id = int(vnf_p_action)

        return Action(vnf_id, srv_id)


def train():
    pass


def test():
    pass
