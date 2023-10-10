from dataclasses import dataclass
from typing import List, Union

from app.types import State, Action
from app.agents.agent import Agent
from app.dl.models.ppo import PPOValue, PPOValueInfo, PPOPolicy, PPOPolicyInfo
from app.dl.models.core import StateEncoder, StateEncoderInfo
from app.utils import utils


@dataclass
class PPOAgentInfo:
    encoder_info: StateEncoderInfo
    vnf_s_policy_info: PPOPolicyInfo
    vnf_p_policy_info: PPOPolicyInfo
    vnf_value_info: PPOValueInfo


class PPOAgent(Agent):
    name = "DRL(PPO)"

    def __init__(self, info: PPOAgentInfo):
        self.info = info

        self.encoder = StateEncoder(info.encoder_info)
        self.vnf_s_policy = PPOPolicy(info.vnf_s_policy_info)
        self.vnf_p_policy = PPOPolicy(info.vnf_p_policy_info)
        self.vnf_value = PPOValue(info.vnf_value_info)

    def inference(self, input: Union[List[List[State]], List[State], State]) -> Action:
        self.encoder.eval()
        self.vnf_s_policy.eval()
        self.vnf_p_policy.eval()
        self.vnf_value.eval()

        rack_x, srv_x, sfc_x, vnf_x, core_x = self.encoder(input)
        vnf_s_out = self.vnf_s_policy(
            core_x.unsqueeze(1).repeat(1, vnf_x.shape[1], 1),
            vnf_x,
            vnf_x.clone(),
        ).unsqueeze(0)
        vnf_s_action, _, _ = utils.get_info_from_logits(vnf_s_out)
        vnf_id = int(vnf_s_action)
        vnf_p_out = self.vnf_p_policy(
            vnf_s_out.unsqueeze(1).repeat(1, srv_x.shape[1], 1),
            srv_x,
            srv_x.clone(),
        ).unsqueeze(0)
        vnf_p_action, _, _ = utils.get_info_from_logits(vnf_p_out)
        srv_id = int(vnf_p_action)
        return Action(vnf_id, srv_id)
