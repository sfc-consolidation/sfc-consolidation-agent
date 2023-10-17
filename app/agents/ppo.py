from dataclasses import dataclass
from typing import List, Union

from app.types import State, Action
from app.agents.agent import Agent
from app.dl.models.ppo import PPOValue, PPOValueInfo, PPOPolicy, PPOPolicyInfo
from app.dl.models.core import StateEncoder, StateEncoderInfo
from app.utils import utils
from app.constants import *

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

    def inference(self, input: Union[List[List[State]], List[State], State]) -> Union[Action, List[Action]]:
        if isinstance(input, State):
            input = [[input]]
        elif isinstance(input[0], State):
            input = [[state] for state in input]
        action_mask = utils.get_possible_action_mask(input).to(TORCH_DEVICE)

        self.encoder.eval()
        self.vnf_s_policy.eval()
        self.vnf_p_policy.eval()
        self.vnf_value.eval()

        rack_x, srv_x, sfc_x, vnf_x, core_x = self.encoder(input)
        vnf_s_out = self.vnf_s_policy(
            core_x.unsqueeze(1).repeat(1, vnf_x.shape[1], 1),
            vnf_x,
            vnf_x.clone(),
        )
        vnf_s_mask = action_mask.sum(dim=2) == 0
        vnf_s_out = vnf_s_out.masked_fill(vnf_s_mask, 0)

        vnf_idxs, _, _ = utils.get_info_from_logits(vnf_s_out)
        
        vnf_p_out = self.vnf_p_policy(
            vnf_s_out.unsqueeze(1).repeat(1, srv_x.shape[1], 1),
            srv_x,
            srv_x.clone(),
        )

        vnf_p_mask = action_mask[torch.arange(vnf_idxs.shape[0]), vnf_idxs, :] == 0
        vnf_p_out = vnf_p_out.masked_fill(vnf_p_mask, 0)

        srv_idxs, _, _ = utils.get_info_from_logits(vnf_p_out)
        
        actions = [
            Action(
                vnfId=input[idx][-1].vnfList[vnf_idx].id,
                srvId=input[idx][-1].vnfList[vnf_idx].srvList[srv_idx].id,
            ) for idx, (vnf_idx, srv_idx) in enumerate(zip(vnf_idxs, srv_idxs))
        ]

        if len(actions) == 1:
            return actions[0]

        return actions