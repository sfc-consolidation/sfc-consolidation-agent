from typing import List

from app.types import State, Action
from app.agents.ppo import PPOAgent, PPOAgentInfo
from app.dl.models.ppo import PPOValueInfo, PPOPolicyInfo
from app.dl.models.core import StateEncoderInfo
from app.dl.models.core import EncoderInfo
from app.constants import *
from app.k8s.envManager import EnvManager
from app.trainers.env import MultiprocessEnvironment
from app.trainers.memory import EpisodeMemory
from app.utils import utils

def live_train(
        env_manager: EnvManager, ppoAgent: PPOAgent,
        encoder_lr: float = 0.01,
        vnf_value_lr: float = 0.01, vnf_value_clip_range: float = float('inf'), vnf_value_max_grad_norm: float = float('inf'),
        vnf_s_policy_lr: float = 0.01, vnf_s_policy_clip_range: float = float('inf'), vnf_s_policy_max_grad_norm: float = float('inf'), vnf_s_entropy_loss_weight: float = float('inf'),
        vnf_p_policy_lr: float = 0.01, vnf_p_policy_clip_range: float = float('inf'), vnf_p_policy_max_grad_norm: float = float('inf'), vnf_p_entropy_loss_weight: float = float('inf'),
        tot_episode_num: int = 100000, gamma: float = 0.99, tau: float = 0.97):
    # 1. setup env
    # 2. setup optimizers and loss_fn
    encoder_optimizer = torch.optim.Adam(ppoAgent.encoder.parameters(), lr=encoder_lr)
    vnf_value_optimizer = torch.optim.Adam(ppoAgent.vnf_value.parameters(), lr=vnf_value_lr)
    vnf_s_policy_optimizer = torch.optim.Adam(ppoAgent.vnf_s_policy.parameters(), lr=vnf_s_policy_lr)
    vnf_p_policy_optimizer = torch.optim.Adam(ppoAgent.vnf_p_policy.parameters(), lr=vnf_p_policy_lr)

    # 3. setup replay memory
    n_workers = 8
    batch_size = 32
    seq_len = 5
    episode_num = 1000
    epochs = 100
    max_episode_len = MAX_EPISODE_LEN
    mp_env = MultiprocessEnvironment(n_workers, env_manager.create_env)
    episode_memory = EpisodeMemory(mp_env, n_workers, batch_size, seq_len, gamma, tau, episode_num, max_episode_len)
    
    # 5. run live training
    for episode in range(tot_episode_num // episode_num):
        episode_memory.fill(ppo_agent)
        for _ in range(epochs):
            update_policy(encoder_optimizer, vnf_s_policy_optimizer, vnf_p_policy_optimizer, ppo_agent, *episode_memory.sample())
        for _ in range(epochs):
            update_value(vnf_value_optimizer, ppo_agent, *episode_memory.sample())
        episode_memory.reset()
    mp_env.close()

def pre_train(): pass

def test(): pass

def update_policy(encoder_optimizer, vnf_s_policy_optimizer, vnf_p_policy_optimizer, ppo_agent: PPOAgent, states: List[List[State]], actions: List[List[Action]], returns: torch.Tensor, gaes: torch.Tensor, vnf_s_logpas: torch.Tensor, vnf_p_logpas: torch.Tensor, values: torch.Tensor) -> None:
    clip_range = 0.1
    entropy_loss_weight = 0.01
    policy_max_grad_norm = float('inf')

    rack_x, srv_x, sfc_x, vnf_x, core_x = ppo_agent.encoder(states)
    vnf_s_outs = ppo_agent.vnf_s_policy(
        core_x.unsqueeze(1).repeat(1, vnf_x.shape[1], 1),
        vnf_x,
        vnf_x.clone(),
    )

    # vnf_s_probs = utils.logit_to_prob(vnf_s_outs)
    # vnf_s_dist = torch.distributions.Categorical(probs=vnf_s_probs)
    # vnf_s_logpas_pred = vnf_s_dist.log_prob([action.vnfId for action in actions])
    # vnf_s_entropies_pred = vnf_s_dist.entropy()

    # vnf_s_rations = torch.exp(vnf_s_logpas_pred - vnf_s_logpas)
    # vnf_s_pi_obj = vnf_s_rations * gaes
    # vnf_s_pi_obj_clipped = vnf_s_rations.clamp(1.0 - clip_range, 1.0 + clip_range) * gaes

    # vnf_s_policy_loss = -torch.min(vnf_s_pi_obj, vnf_s_pi_obj_clipped).mean()
    # vnf_s_entropy_loss = -vnf_s_entropies_pred.mean()
    # vnf_s_policy_loss = vnf_s_policy_loss + entropy_loss_weight * vnf_s_entropy_loss

    

    vnf_p_outs = ppo_agent.vnf_p_policy(
        vnf_s_outs.unsqueeze(1).repeat(1, srv_x.shape[1], 1),
        srv_x,
        srv_x.clone(),
    )
    vnf_p_probs = utils.logit_to_prob(vnf_p_outs)
    vnf_p_dist = torch.distributions.Categorical(probs=vnf_p_probs)
    vnf_p_logpas_pred = vnf_p_dist.log_prob([action.srvId for action in actions])
    vnf_p_entropies_pred = vnf_p_dist.entropy()

    vnf_p_rations = torch.exp(vnf_p_logpas_pred - vnf_p_logpas)
    vnf_p_pi_obj = vnf_p_rations * gaes[:, 1]
    vnf_p_pi_obj_clipped = vnf_p_rations.clamp(1.0 - clip_range, 1.0 + clip_range) * gaes

    vnf_p_policy_loss = -torch.min(vnf_p_pi_obj, vnf_p_pi_obj_clipped).mean()
    vnf_p_entropy_loss = -vnf_p_entropies_pred.mean()
    vnf_p_policy_loss = vnf_p_policy_loss + entropy_loss_weight * vnf_p_entropy_loss

    encoder_optimizer.zero_grad()
    vnf_p_policy_optimizer.zero_grad()
    vnf_p_policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(ppo_agent.vnf_p_policy.parameters(), policy_max_grad_norm)
    vnf_p_policy_optimizer.step()
    encoder_optimizer.step()


def update_value(value_optimizer, ppo_agent: PPOAgent, states: List[List[State]], actions: List[List[Action]], returns: torch.Tensor, gaes: torch.Tensor, vnf_s_logpas: torch.Tensor, vnf_p_logpas: torch.Tensor, values: torch.Tensor) -> None:
    clip_range = float('inf')
    max_grad_norm = float('inf')
    vnf_s_value_pred = ppo_agent.vnf_value(states)
    vnf_s_value_pred_clipped = values + (vnf_s_value_pred - values).clamp(-clip_range, clip_range)

    vnf_s_value_loss = (vnf_s_value_pred - returns).pow(2)
    vnf_s_value_loss_clipped = (vnf_s_value_pred_clipped - returns).pow(2)
    vnf_s_value_loss = 0.5 * torch.max(vnf_s_value_loss, vnf_s_value_loss_clipped).mean()

    value_optimizer.zero_grad()
    vnf_s_value_loss.backward()
    torch.nn.utils.clip_grad_norm_(ppo_agent.vnf_value.parameters(), max_grad_norm)
    value_optimizer.step()

stateEncoderInfo = StateEncoderInfo(
    max_rack_num=MAX_RACK_NUM,
    rack_id_dim=2,
    max_srv_num=MAX_SRV_NUM,
    srv_id_dim=2,
    srv_encoder_info=EncoderInfo(
        input_size=2 + 3,
        output_size=4,
        hidden_sizes=[8],
        batch_norm=True,
        method="SA",
        dropout=0.3,
        num_head=2,
        device=TORCH_DEVICE,
    ),
    max_sfc_num=MAX_SFC_NUM,
    sfc_id_dim=4,
    sfc_encoder_info=EncoderInfo(
        input_size=4 + 1,
        output_size=4,
        hidden_sizes=[8],
        batch_norm=True,
        method="SA",
        dropout=0.3,
        num_head=2,
        device=TORCH_DEVICE,
    ),
    max_vnf_num=MAX_VNF_NUM,
    vnf_id_dim=4,
    vnf_encoder_info=EncoderInfo(
        input_size=4 + 2 + 4 + 4 + 3,
        output_size=8,
        hidden_sizes=[16],
        batch_norm=True,
        method="SA",
        dropout=0.3,
        num_head=4,
        device=TORCH_DEVICE,
    ),
    core_encoder_info=EncoderInfo(
        input_size=2 * MAX_RACK_NUM + 4 * MAX_SRV_NUM + 4 * MAX_SFC_NUM + 8 * MAX_VNF_NUM,
        output_size=8,
        hidden_sizes=[32, 16],
        batch_norm=True,
        method="LSTM",
        dropout=0.3,
        device=TORCH_DEVICE,
    ),
    device=TORCH_DEVICE,
)

if __name__ == "__main__":
    ppo_agent = PPOAgent(PPOAgentInfo(
        encoder_info=stateEncoderInfo,
        vnf_value_info=PPOValueInfo(
            query_size=8,
            key_size=8,
            value_size=8,
            hidden_sizes=[8, 8],
            num_heads=[4, 4],
            dropout=0.3,
            seq_len=MAX_VNF_NUM,
            device=TORCH_DEVICE,
        ),
        vnf_p_policy_info=PPOPolicyInfo(
            query_size=8,
            key_size=8,
            value_size=8,
            hidden_sizes=[8, 8],
            num_heads=[4, 4],
            dropout=0.3,
            device=TORCH_DEVICE,
        ),
        vnf_s_policy_info=PPOPolicyInfo(
            query_size=8,
            key_size=8,
            value_size=8,
            hidden_sizes=[8, 8],
            num_heads=[4, 4],
            dropout=0.3,
            device=TORCH_DEVICE,
        ),
    ))

    env_manager = EnvManager()
    try:
        env_manager.delete_all()
        live_train(env_manager, ppo_agent)
    finally:
        env_manager.delete_all()