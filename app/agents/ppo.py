from dataclasses import dataclass
from typing import List, Union

from app.types import State, Action
from app.agents.agent import Agent
from app.dl.models.ppo import PPOValue, PPOValueInfo, PPOPolicy, PPOPolicyInfo
from app.dl.models.core import StateEncoder, StateEncoderInfo, Stopper, StopperInfo
from app.utils import utils


@dataclass
class PPOAgentInfo:
    encoder_info: StateEncoderInfo
    stopper_info: StopperInfo
    vnf_s_value_info: PPOValueInfo
    vnf_s_policy_info: PPOPolicyInfo
    vnf_p_value_info: PPOValueInfo
    vnf_p_policy_info: PPOPolicyInfo


class PPOAgent(Agent):
    name = "DRL(PPO)"

    def __init__(self, info: PPOAgentInfo):
        self.info = info

        self.encoder = StateEncoder(info.encoder_info)
        self.stopper = Stopper(info.stopper_info)
        self.vnf_s_value = PPOValue(info.vnf_s_value_info)
        self.vnf_s_policy = PPOPolicy(info.vnf_s_policy_info)
        self.vnf_p_value = PPOValue(info.vnf_p_value_info)
        self.vnf_p_policy = PPOPolicy(info.vnf_p_policy_info)

    def inference(self, input: Union[List[List[State]], List[State], State]) -> Action:
        self.encoder.eval()
        self.stopper.eval()
        self.vnf_s_value.eval()
        self.vnf_s_policy.eval()
        self.vnf_p_value.eval()
        self.vnf_p_policy.eval()

        rack_x, srv_x, sfc_x, vnf_x, core_x = self.encoder(input)
        stop = self.stopper(core_x)
        if (stop > 0.5):
            return None
        vnf_s_out = self.vnf_s_policy(
            core_x.repeat(1, vnf_x.shape[1], 1),
            vnf_x,
            vnf_x.clone(),
        ).unsqueeze(0)
        vnf_s_action, _, _ = utils.get_info_from_logits(vnf_s_out)
        vnf_id = int(vnf_s_action)
        vnf_p_out = self.vnf_p_policy(
            vnf_s_out.repeat(1, srv_x.shape[1], 1),
            srv_x,
            srv_x.clone(),
        ).unsqueeze(0)
        vnf_p_action, _, _ = utils.get_info_from_logits(vnf_p_out)
        srv_id = int(vnf_p_action)
        return Action(vnf_id, srv_id)


def live_train():
    pass


def pre_train():
    pass


def test():
    pass
