from typing import List
import asyncio

from app.types import State, Action
from app.agents.ppo import PPOAgent, PPOAgentInfo
from app.dl.models.ppo import PPOValueInfo, PPOPolicyInfo
from app.dl.models.core import StateEncoderInfo
from app.dl.models.core import EncoderInfo
from app.constants import *
from app.k8s.envManager import EnvManager, ResetArg
from app.trainers.env import MultiprocessEnvironment
from app.trainers.memory import EpisodeMemory
from app.utils import utils


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

DROPOUT_RATE = 0.1
GAMMA = 1.0

resetArg = ResetArg(
    maxRackNum=2, minRackNum=2,
    maxSrvNumInSingleRack=3, minSrvNumInSingleRack=3,
    maxVnfNum=10, minVnfNum=10,
    maxSfcNum=3, minSfcNum=3,
    maxSrvVcpuNum=100, minSrvVcpuNum=100,
    maxSrvVmemMb=32 * 1024, minSrvVmemMb=32 * 1024,
    maxVnfVcpuNum=1, minVnfVcpuNum=1,
    maxVnfVmemMb=1024 // 2, minVnfVmemMb=1024 * 4,
)

async def live_train(
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
        await episode_memory.fill(ppo_agent)
        for epoch in range(epochs):
            vnf_s_policy_loss, vnf_p_policy_loss , vnf_policy_loss = update_policy(encoder_optimizer, vnf_s_policy_optimizer, vnf_p_policy_optimizer, ppo_agent, *episode_memory.sample())
            writer.add_scalar("[PPO Live Train] Policy Loss in VNF Selection", vnf_s_policy_loss.item(), episode * epochs + epoch)
            writer.add_scalar("[PPO Live Train] Policy Loss in VNF Placement", vnf_p_policy_loss.item(), episode * epochs + epoch)
            writer.add_scalar("[PPO Live Train] Policy Total Loss", vnf_policy_loss.item(), episode * epochs + epoch)
        for _ in range(epochs):
            vnf_value_loss = update_value(vnf_value_optimizer, ppo_agent, *episode_memory.sample())
            writer.add_scalar("[PPO Live Train] Value Total Loss", vnf_value_loss.item(), episode * epochs + epoch)
            writer.add_scalar("[PPO Live Train] Episode Reward", episode_memory.reward.mean().item(), episode * epochs + epoch)
        episode_memory.reset()

def pre_train(): pass

def test(): pass

def update_policy(encoder_optimizer, vnf_s_policy_optimizer, vnf_p_policy_optimizer, ppo_agent: PPOAgent, states: List[List[State]], actions: List[List[Action]], returns: torch.Tensor, gaes: torch.Tensor, vnf_s_logpas: torch.Tensor, vnf_p_logpas: torch.Tensor, values: torch.Tensor):
    clip_range = 0.1
    entropy_loss_weight = 0.01
    policy_max_grad_norm = float('inf')

    ppo_agent.encoder.train()
    ppo_agent.vnf_s_policy.train()
    ppo_agent.vnf_p_policy.train()

    rack_x, srv_x, sfc_x, vnf_x, core_x = ppo_agent.encoder(states)
    vnf_s_outs = ppo_agent.vnf_s_policy(
        core_x.unsqueeze(1).repeat(1, vnf_x.shape[1], 1),
        vnf_x,
        vnf_x.clone(),
    )

    vnf_s_probs = utils.logit_to_prob(vnf_s_outs)
    vnf_s_dist = torch.distributions.Categorical(probs=vnf_s_probs)
    vnf_s_logpas_pred = vnf_s_dist.log_prob(torch.tensor([action_seq[-1].vnfId for action_seq in actions]).to(TORCH_DEVICE))
    vnf_s_entropies_pred = vnf_s_dist.entropy()

    vnf_s_rations = torch.exp(vnf_s_logpas_pred - vnf_s_logpas.to(TORCH_DEVICE))
    vnf_s_pi_obj = vnf_s_rations * gaes.to(TORCH_DEVICE)
    vnf_s_pi_obj_clipped = vnf_s_rations.clamp(1.0 - clip_range, 1.0 + clip_range) * gaes.to(TORCH_DEVICE)

    vnf_s_policy_loss = -torch.min(vnf_s_pi_obj, vnf_s_pi_obj_clipped).mean()
    vnf_s_entropy_loss = -vnf_s_entropies_pred.mean()
    vnf_s_policy_loss = vnf_s_policy_loss + entropy_loss_weight * vnf_s_entropy_loss

    vnf_p_outs = ppo_agent.vnf_p_policy(
        vnf_s_outs.unsqueeze(1).repeat(1, srv_x.shape[1], 1),
        srv_x,
        srv_x.clone(),
    )
    vnf_p_probs = utils.logit_to_prob(vnf_p_outs)
    vnf_p_dist = torch.distributions.Categorical(probs=vnf_p_probs)
    vnf_p_logpas_pred = vnf_p_dist.log_prob(torch.tensor([action_seq[-1].srvId for action_seq in actions]).to(TORCH_DEVICE))
    vnf_p_entropies_pred = vnf_p_dist.entropy()

    vnf_p_rations = torch.exp(vnf_p_logpas_pred - vnf_p_logpas.to(TORCH_DEVICE))
    vnf_p_pi_obj = vnf_p_rations * gaes.to(TORCH_DEVICE)
    vnf_p_pi_obj_clipped = vnf_p_rations.clamp(1.0 - clip_range, 1.0 + clip_range) * gaes.to(TORCH_DEVICE)

    vnf_p_policy_loss = -torch.min(vnf_p_pi_obj, vnf_p_pi_obj_clipped).mean().to(TORCH_DEVICE)
    vnf_p_entropy_loss = -vnf_p_entropies_pred.mean().to(TORCH_DEVICE)
    vnf_p_policy_loss = vnf_p_policy_loss + entropy_loss_weight * vnf_p_entropy_loss

    vnf_loss = vnf_s_policy_loss + vnf_p_policy_loss

    encoder_optimizer.zero_grad()
    vnf_s_policy_optimizer.zero_grad()
    vnf_p_policy_optimizer.zero_grad()
    vnf_loss.backward()
    torch.nn.utils.clip_grad_norm_(ppo_agent.encoder.parameters(), policy_max_grad_norm)
    torch.nn.utils.clip_grad_norm_(ppo_agent.vnf_s_policy.parameters(), policy_max_grad_norm)
    torch.nn.utils.clip_grad_norm_(ppo_agent.vnf_p_policy.parameters(), policy_max_grad_norm)
    encoder_optimizer.step()
    vnf_s_policy_optimizer.step()
    vnf_p_policy_optimizer.step()

    return vnf_s_policy_loss, vnf_p_policy_loss, vnf_loss

def update_value(value_optimizer, ppo_agent: PPOAgent, states: List[List[State]], actions: List[List[Action]], returns: torch.Tensor, gaes: torch.Tensor, vnf_s_logpas: torch.Tensor, vnf_p_logpas: torch.Tensor, values: torch.Tensor):
    clip_range = float('inf')
    max_grad_norm = float('inf')

    ppo_agent.vnf_value.train()

    rack_x, srv_x, sfc_x, vnf_x, core_x = ppo_agent.encoder(states)
    value_pred = ppo_agent.vnf_value(core_x)
    vnf_s_value_pred_clipped = values + (value_pred - values).clamp(-clip_range, clip_range)

    vnf_s_value_loss = (value_pred - returns).pow(2)
    vnf_s_value_loss_clipped = (vnf_s_value_pred_clipped - returns).pow(2)
    vnf_s_value_loss = 0.5 * torch.max(vnf_s_value_loss, vnf_s_value_loss_clipped).mean()

    value_optimizer.zero_grad()
    vnf_s_value_loss.backward()
    torch.nn.utils.clip_grad_norm_(ppo_agent.vnf_value.parameters(), max_grad_norm)
    value_optimizer.step()

    return vnf_s_value_loss

stateEncoderInfo = StateEncoderInfo(
    max_rack_num=MAX_RACK_NUM,
    rack_id_dim=2,
    max_srv_num=MAX_SRV_NUM,
    srv_id_dim=2,
    srv_encoder_info=EncoderInfo(
        input_size=2 + 2 + 3,
        output_size=4,
        hidden_sizes=[8],
        batch_norm=True,
        method="SA",
        dropout=DROPOUT_RATE,
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
        dropout=DROPOUT_RATE,
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
        dropout=DROPOUT_RATE,
        num_head=4,
        device=TORCH_DEVICE,
    ),
    core_encoder_info=EncoderInfo(
        input_size=2 * MAX_RACK_NUM + 4 * MAX_SRV_NUM + 4 * MAX_SFC_NUM + 8 * MAX_VNF_NUM,
        output_size=8,
        hidden_sizes=[32, 16],
        batch_norm=True,
        method="LSTM",
        dropout=DROPOUT_RATE,
        device=TORCH_DEVICE,
    ),
    device=TORCH_DEVICE,
)

if __name__ == "__main__":
    ppo_agent = PPOAgent(PPOAgentInfo(
        encoder_info=stateEncoderInfo,
        vnf_value_info=PPOValueInfo(
            input_size=8,
            hidden_sizes=[8, 8],
            dropout=DROPOUT_RATE,
            device=TORCH_DEVICE,
        ),
        vnf_s_policy_info=PPOPolicyInfo(
            query_size=8,
            key_size=8,
            value_size=8,
            hidden_sizes=[8, 8],
            num_heads=[4, 4],
            dropout=DROPOUT_RATE,
            device=TORCH_DEVICE,
        ),
        vnf_p_policy_info=PPOPolicyInfo(
            query_size=MAX_VNF_NUM,
            key_size=4,
            value_size=4,
            hidden_sizes=[16, 16],
            num_heads=[4, 4],
            dropout=DROPOUT_RATE,
            device=TORCH_DEVICE,
        ),
    ))

    env_manager = EnvManager()
    try:
        env_manager.delete_all()
        loop = asyncio.get_event_loop()
        tasks = [ loop.create_task(live_train(env_manager, ppo_agent, gamma=GAMMA)) ]
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    finally:
        env_manager.delete_all()