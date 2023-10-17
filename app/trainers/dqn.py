import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from typing import List

from app.agents.dqn import DQNAgent, DQNAgentInfo
from app.dl.models.encoder import EncoderInfo
from app.dl.models.core import StateEncoderInfo
from app.dl.models.dqn import DQNAdvantageInfo, DQNValueInfo
from app.agents.random import RandomAgent
from app.utils import utils

from app.agents.dqn import DQNAgent
from app.agents.random import RandomAgent
from app.trainers.env import Environment
from app.k8s.envManager import EnvManager, ResetArg
from app.constants import *
from app.trainers.memory import ReplayMemory, PrioritizedReplayMemory, Data
from app.trainers.debug import Debugger
from app.types import Episode, Action

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


DROPOUT_RATE = 0.1
GAMMA = 1.0

resetArg = ResetArg(
    maxRackNum=2, minRackNum=2,
    maxSrvNumInSingleRack=3, minSrvNumInSingleRack=3,
    maxVnfNum=10, minVnfNum=10,
    maxSfcNum=1, minSfcNum=1,
    maxSrvVcpuNum=100, minSrvVcpuNum=100,
    maxSrvVmemMb=1024, minSrvVmemMb=1024,
    maxVnfVcpuNum=1, minVnfVcpuNum=1,
    maxVnfVmemMb=512, minVnfVmemMb=512,
)

# Rainbow (1) DQN
def update_main(dqn_agent: DQNAgent, memory, encoder_optimizer, value_s_optimizer, value_p_optimizer, vnf_s_optimizer, vnf_p_optimizer, gamma, beta = None):
    if beta is not None:
        batch, weights, sample_idxs = memory.sample(beta=beta)
    else:
        batch = memory.sample()
    batch_size = len(batch)

    # State -> (Batch, Seq, 1)
    state_seq_batch = [[sample.state if sample != None else None for sample in seq] for seq in batch]
    # Action -> (Batch, 1)
    vnf_s_action_batch = torch.tensor([seq[-1].action.vnfId for seq in batch]).to(TORCH_DEVICE)
    # Action -> (Batch, 1)
    vnf_p_action_batch = torch.tensor([seq[-1].action.srvId for seq in batch]).to(TORCH_DEVICE)
    # Done -> (Batch, 1)
    done_batch = torch.tensor([seq[-1].done for seq in batch]).to(torch.float32).to(TORCH_DEVICE)
    # Reward -> (Batch, 1)
    reward_batch = torch.tensor([seq[-1].reward for seq in batch]).to(TORCH_DEVICE)
    # Next State -> (Batch, Seq, 1)
    next_state_seq_batch = []
    for seq in batch:
        next_state_seq = []
        for sample in seq:
            if sample == None:
                next_state_seq.append(None)
            else:
                # Next State seq는 기본적으로 State seq보다 하나 적은 None을 가진다.
                if not len(next_state_seq) == 0:
                    next_state_seq.pop()
                    next_state_seq.append(sample.state)
                next_state_seq.append(sample.next_state)
        next_state_seq_batch.append(next_state_seq)

    dqn_agent.encoder.train()

    rack_x, srv_x, sfc_x, vnf_x, core_x = dqn_agent.encoder(state_seq_batch)

    srv_x = srv_x.detach()
    vnf_x = vnf_x.detach()

    next_rack_x, next_srv_x, next_sfc_x, next_vnf_x, next_core_x = dqn_agent.encoder(next_state_seq_batch)

    next_srv_x = next_srv_x.detach()
    next_vnf_x = next_vnf_x.detach()
    next_core_x = next_core_x.detach()

    dqn_agent.vnf_s_advantage.train()
    dqn_agent.vnf_s_value.train()
    
    action_mask = utils.get_possible_action_mask(state_seq_batch).to(TORCH_DEVICE)
    vnf_s_mask = action_mask.sum(dim=2) == 0

    vnf_s_advantage = dqn_agent.vnf_s_advantage(
        core_x.unsqueeze(1).repeat(1, vnf_x.shape[1], 1),
        vnf_x,
        vnf_x.clone(),
    )
    vnf_s_advantage = vnf_s_advantage.masked_fill(vnf_s_mask, -1e-9)

    vnf_s_value = dqn_agent.vnf_s_value(core_x)
    # Rainbow (4) Dueling DQN
    vnf_s_q = vnf_s_value + vnf_s_advantage - vnf_s_advantage.mean(dim=-1, keepdim=True)
    vnf_s_q = vnf_s_q.gather(1, vnf_s_action_batch.unsqueeze(1))

    dqn_agent.vnf_s_advantage.eval()
    dqn_agent.vnf_s_value.eval()

    next_vnf_s_advantage = dqn_agent.vnf_s_advantage(
        next_core_x.unsqueeze(1).repeat(1, next_vnf_x.shape[1], 1),
        next_vnf_x,
        next_vnf_x.clone(),
    )
    next_vnf_s_advantage = next_vnf_s_advantage.masked_fill(vnf_s_mask, -1e-9)
    next_vnf_s_value = dqn_agent.vnf_s_value(next_core_x)
    next_vnf_s_q = next_vnf_s_value + next_vnf_s_advantage - next_vnf_s_advantage.mean(dim=-1, keepdim=True)
    # Rainbow (4) Dueling DQN
    vnf_s_expected_q = (reward_batch + gamma * next_vnf_s_q.max(1)[0].detach() * (1 - done_batch)).unsqueeze(1)

    dqn_agent.vnf_p_advantage.train()
    dqn_agent.vnf_p_value.train()

    vnf_p_mask = action_mask[torch.arange(batch_size), vnf_s_action_batch, :] == 0
    
    vnf_p_advantage = dqn_agent.vnf_p_advantage(
        vnf_s_advantage.detach().unsqueeze(1).repeat(1, srv_x.shape[1], 1),
        srv_x,
        srv_x.clone(),
    )
    vnf_p_advantage = vnf_p_advantage.masked_fill(vnf_p_mask, -1e-9)

    vnf_p_value = dqn_agent.vnf_p_value(vnf_s_advantage)
    # Rainbow (4) Dueling DQN
    vnf_p_q = vnf_p_value + vnf_p_advantage - vnf_p_advantage.mean(dim=-1, keepdim=True)
    vnf_p_q = vnf_p_q.gather(1, vnf_p_action_batch.unsqueeze(1))
    
    dqn_agent.vnf_p_advantage.eval()
    dqn_agent.vnf_p_value.eval()

    next_vnf_p_advantage = dqn_agent.vnf_p_advantage(
        next_vnf_s_advantage.detach().unsqueeze(1).repeat(1, next_srv_x.shape[1], 1),
        next_srv_x,
        next_srv_x.clone(),
    )
    next_vnf_p_advantage = next_vnf_p_advantage.masked_fill(vnf_p_mask, -1e-9)

    next_vnf_p_value = dqn_agent.vnf_p_value(next_vnf_s_advantage)
    # Rainbow (4) Dueling DQN
    next_vnf_p_q = next_vnf_p_value + next_vnf_p_advantage - next_vnf_p_advantage.mean(dim=-1, keepdim=True)

    vnf_p_expected_q = (reward_batch + gamma * next_vnf_p_q.max(1)[0].detach() * (1 - done_batch)).unsqueeze(1)
    
    # Rainbow (3) Prioritized Experience Replay
    if beta is not None:
        elementwise_vnf_s_loss = F.smooth_l1_loss(vnf_s_q, vnf_s_expected_q, reduction="none")
        elementwise_vnf_p_loss = F.smooth_l1_loss(vnf_p_q, vnf_p_expected_q, reduction="none")
        elementwise_vnf_loss = elementwise_vnf_s_loss + elementwise_vnf_p_loss
        weights = torch.FloatTensor(weights).to(TORCH_DEVICE)
        vnf_s_loss = torch.mean(elementwise_vnf_s_loss * weights)
        vnf_p_loss = torch.mean(elementwise_vnf_p_loss * weights)
        vnf_loss = torch.mean(elementwise_vnf_loss * weights)
        loss_for_prior = elementwise_vnf_loss.detach().cpu().numpy() # Rainbow (3) Prioritized Experience Replay
        new_priorities = loss_for_prior + EPS # Rainbow (3) Prioritized Experience Replay
        memory.update_priorities(sample_idxs, new_priorities) # Rainbow (3) Prioritized Experience Replay
    else:
        vnf_s_loss = F.smooth_l1_loss(vnf_s_q, vnf_s_expected_q)
        vnf_p_loss = F.smooth_l1_loss(vnf_p_q, vnf_p_expected_q)
        vnf_loss = vnf_s_loss + vnf_p_loss
    encoder_optimizer.zero_grad()
    vnf_s_optimizer.zero_grad()
    vnf_p_optimizer.zero_grad()
    vnf_loss.backward()
    torch.nn.utils.clip_grad_norm_(main_agent.vnf_s_advantage.parameters(), 1) # Rainbow (0) Gradient Clipping 
    torch.nn.utils.clip_grad_norm_(main_agent.encoder.parameters(), 1) # Rainbow (0) Gradient Clipping
    torch.nn.utils.clip_grad_norm_(main_agent.vnf_p_advantage.parameters(), 1) # Rainbow (0) Gradient Clipping 
    encoder_optimizer.step()
    vnf_s_optimizer.step()
    vnf_p_optimizer.step()

    return vnf_s_loss, vnf_p_loss, vnf_loss, reward_batch.mean()

# Rainbow (2) Double DQN
def update_target(main_dqn_agent: DQNAgent, target_dqn_agent: DQNAgent):
    target_dqn_agent.encoder.load_state_dict(main_dqn_agent.encoder.state_dict())
    target_dqn_agent.vnf_s_advantage.load_state_dict(main_dqn_agent.vnf_s_advantage.state_dict())
    target_dqn_agent.vnf_p_advantage.load_state_dict(main_dqn_agent.vnf_p_advantage.state_dict())

def live_train(env_manager: EnvManager, main_agent: DQNAgent, target_agent: DQNAgent, encoder_lr: float, vnf_s_lr: float, vnf_p_lr: float, tot_episode_num: int, gamma: float, alpha: float, beta: float):
    # 1. setup env
    env = env_manager.create_env(id="dqn-live-train")
    random_agent = RandomAgent()

    # 2. setup optimizers
    encoder_optimizer = torch.optim.Adam(main_agent.encoder.parameters(), lr=encoder_lr)
    vnf_s_value_optimizer = torch.optim.Adam(main_agent.vnf_s_value.parameters(), lr=vnf_s_lr)
    vnf_p_value_optimizer = torch.optim.Adam(main_agent.vnf_p_value.parameters(), lr=vnf_p_lr)
    vnf_s_advantage_optimizer = torch.optim.Adam(main_agent.vnf_s_advantage.parameters(), lr=vnf_s_lr)
    vnf_p_advantage_optimizer = torch.optim.Adam(main_agent.vnf_p_advantage.parameters(), lr=vnf_p_lr)
    

    # 3. setup replay memory
    batch_size = 16
    seq_len = 5
    memory = PrioritizedReplayMemory(alpha, batch_size, seq_len, 1_000, 4, gamma)

    # 4. set debugging pockets
    debugger = Debugger()

    # 5. run live training
    # - exploration을 조금씩 줄여나가면서 업데이트 진행
    # - run dqn three time
    # 에피소드를 전체 한 번에 받고,
    # 해당 episode를 활용해서 update하기
    epsilon = 0.5
    for episode_num in range(1, tot_episode_num + 1):
        if episode_num % 10 == 0:
            update_target(main_agent, target_agent)
        state, info, done = env.reset(resetArg)
        ini_state = deepcopy(state)
        ini_info = deepcopy(info)
        
        # Epsilon Greedy
        explore_rate = epsilon * (1 - episode_num / tot_episode_num)

        # Rainbow (3) Prioritized Experience Replay
        fraction = min(episode_num / tot_episode_num, 1.0)
        beta = beta + fraction * (1.0 - beta)

        history = []
        for step_num in range(1, 11):
            history.append(state)
            if np.random.uniform() < explore_rate:
                # random
                action = random_agent.inference(state)
            else:
                # greedy
                action = target_agent.inference([history[max(0, len(history) - seq_len):len(history)]])
            next_state, next_info, done = env.step(action)
            if done:
                break

            memory.append(Data(
                episode_num=episode_num,
                step_num=step_num,
                state=state,
                action=action,
                reward=utils.calc_reward(info, next_info),
                next_state=next_state,
                done=done,
            ))
            
            state = next_state
            info = next_info
            if len(memory) < batch_size:
                continue
            vnf_s_loss, vnf_p_loss, loss, reward = update_main(main_agent, memory, encoder_optimizer, vnf_s_value_optimizer, vnf_p_value_optimizer, vnf_s_advantage_optimizer, vnf_p_advantage_optimizer, gamma, beta)

            writer.add_scalar("[Live Train] Loss in VNF Selection", vnf_s_loss.item(), (episode_num - 1) * 10 + step_num)
            writer.add_scalar("[Live Train] Loss in VNF Placement", vnf_p_loss.item(), (episode_num - 1) * 10 + step_num)
            writer.add_scalar("[Live Train] Total Loss", loss.item(), (episode_num - 1) * 10 + step_num)
            writer.add_scalar("[Live Train] Reward Mean", reward.item(), (episode_num - 1) * 10 + step_num)

        fin_state = deepcopy(state)
        fin_info = deepcopy(info)

        debugger.add_episode(ini_state, ini_info, fin_state, fin_info, explore_rate, step_num)
        debugger.print(last_n=100)

def pre_train(env_manager: EnvManager, dqn_agent: DQNAgent, encoder_lr: float, vnf_s_lr: float, vnf_p_lr: float, tot_episode_num: int, gamma: float):
    # 1. setup env
    env = env_manager.create_env(id="dqn-pre-train")
    debugger = Debugger()

    # 2. setup optimizers
    encoder_optimizer = torch.optim.Adam(dqn_agent.encoder.parameters(), lr=encoder_lr)
    vnf_s_optimizer = torch.optim.Adam(dqn_agent.vnf_s_advantage.parameters(), lr=vnf_s_lr)
    vnf_p_optimizer = torch.optim.Adam(dqn_agent.vnf_p_advantage.parameters(), lr=vnf_p_lr)

    # 3. setup replay memory
    batch_size = 16
    seq_len = 5
    memory = ReplayMemory(batch_size, seq_len, 1_000_000, n_step=1, gamma=gamma)

    pre_data_path = "./data/episode/ff"
    # get all filename in pre_data_path
    pre_data_list = os.listdir(pre_data_path)
    episode_num = 0
    # read all data
    for pre_data in pre_data_list:
        episode_num += 1
        path = os.path.join(pre_data_path, pre_data)
        with open(path, "r") as f:
            episode = json.load(f)
            episode = utils.dataclass_from_dict(Episode, episode)
            stateList = list(map(lambda step: step.state, episode.steps))
            actionList = list(map(lambda step: Action(step.action["vnfId"], step.action["srvId"]) if step.action is not None else None, episode.steps))
            infoList = list(map(lambda step: step.info, episode.steps))
            
            for i in range(len(stateList) - 1):
                done = not infoList[i+1].success
                if done: break
                memory.append(Data(
                    episode_num=episode_num, 
                    step_num=i,
                    state=stateList[i],
                    action=actionList[i],
                    next_state=stateList[i+1],
                    reward=utils.calc_reward(infoList[i], infoList[i+1]),
                    done=done,
                ))
        print(f"[PreTrain] {episode_num} episodes loaded")

    # 4. run live training
    # - exploration을 조금씩 줄여나가면서 업데이트 진행
    # - run dqn three time
    # 에피소드를 전체 한 번에 받고,
    # 해당 episode를 활용해서 update하기
    for episode_num in range(1, tot_episode_num + 1):
        if len(memory) < batch_size:
            continue
        vnf_s_loss, vnf_p_loss, loss, reward = update_main(dqn_agent, memory, encoder_optimizer, vnf_s_optimizer, vnf_p_optimizer, gamma)
        writer.add_scalar("[Pre Train] Loss in VNF Selection", vnf_s_loss, episode_num)
        writer.add_scalar("[Pre Train] Loss in VNF Placement", vnf_p_loss, episode_num)
        writer.add_scalar("[Pre Train] Total Loss ", loss, episode_num)
        writer.add_scalar("[Pre Train] Reward Mean", reward, episode_num)

        if episode_num == 1 or episode_num % 10 == 0:
            test(env, dqn_agent, seq_len, debugger, episode_num)

def test(env, agent, seq_len, debugger, episode_num):
    history = []
    state, info, done = env.reset(resetArg)
    ini_state = deepcopy(state)
    ini_info = deepcopy(info)
    for step_num in range(1, 6):
        history.append(state)
        action = agent.inference([history[max(0, len(history) - seq_len):len(history)]])
        next_state, next_info, done = env.step(action)
        if done:
            break
        state = next_state
        info = next_info
    fin_state = deepcopy(state)
    fin_info = deepcopy(info)
    debugger.add_episode(ini_state, ini_info, fin_state, fin_info, 0, step_num)
    print(f"Episode: {episode_num}")
    debugger.print(last_n=1)



stateEncoderInfo = StateEncoderInfo(
    max_rack_num=MAX_RACK_NUM,
    rack_id_dim=2,
    max_srv_num=MAX_SRV_NUM,
    srv_id_dim=2,
    srv_encoder_info=EncoderInfo(
        input_size=2 + 2 + 3,
        output_size=4,
        hidden_sizes=[8, 8],
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
        hidden_sizes=[8, 8],
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
        hidden_sizes=[16, 16],
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

def ff_test(env_manager: EnvManager):
    env = env_manager.create_env(id="dqn-ff-test")
    from app.agents.ff import FFAgent
    ff_agent = FFAgent()
    state, info, done = env.reset(resetArg)
    ini_state = deepcopy(state)
    ini_info = deepcopy(info)
    debugger = Debugger()
    for step_num in range(1, 6):
        action = ff_agent.inference(state)
        next_state, next_info, done = env.step(action)
        if done:
            break
        state = next_state
        info = next_info
    fin_state = deepcopy(state)
    fin_info = deepcopy(info)
    debugger.add_episode(ini_state, ini_info, fin_state, fin_info, 0, step_num)
    debugger.print(last_n=1)

if __name__ == "__main__":
    main_agent = DQNAgent(DQNAgentInfo(
        encoder_info=stateEncoderInfo,
        vnf_s_advantage_info=DQNAdvantageInfo(
            query_size=8,
            key_size=8,
            value_size=8,
            hidden_sizes=[16, 16],
            num_heads=[4, 4],
            dropout=DROPOUT_RATE,
            device=TORCH_DEVICE,
        ),
        vnf_p_advantage_info=DQNAdvantageInfo(
            query_size=MAX_VNF_NUM,
            key_size=4,
            value_size=4,
            hidden_sizes=[16, 16],
            num_heads=[4, 4],
            dropout=DROPOUT_RATE,
            device=TORCH_DEVICE,
        ),
        vnf_s_value_info=DQNValueInfo(
            input_size=8,
            hidden_sizes=[16, 16],
            dropout=DROPOUT_RATE,
            device=TORCH_DEVICE,
        ),
        vnf_p_value_info=DQNValueInfo(
            input_size=MAX_VNF_NUM,
            hidden_sizes=[16, 16],
            dropout=DROPOUT_RATE,
            device=TORCH_DEVICE,
        ),
    ))
    target_agent = DQNAgent(DQNAgentInfo(
        encoder_info=stateEncoderInfo,
        vnf_s_advantage_info=DQNAdvantageInfo(
            query_size=8,
            key_size=8,
            value_size=8,
            hidden_sizes=[16, 16],
            num_heads=[4, 4],
            dropout=DROPOUT_RATE,
            device=TORCH_DEVICE,
        ),
        vnf_p_advantage_info=DQNAdvantageInfo(
            query_size=MAX_VNF_NUM,
            key_size=4,
            value_size=4,
            hidden_sizes=[16, 16],
            num_heads=[4, 4],
            dropout=DROPOUT_RATE,
            device=TORCH_DEVICE,
        ),
        vnf_s_value_info=DQNValueInfo(
            input_size=8,
            hidden_sizes=[16, 16],
            dropout=DROPOUT_RATE,
            device=TORCH_DEVICE,
        ),
        vnf_p_value_info=DQNValueInfo(
            input_size=MAX_VNF_NUM,
            hidden_sizes=[16, 16],
            dropout=DROPOUT_RATE,
            device=TORCH_DEVICE,
        ),
    ))
    env_manager = EnvManager()

    alpha= 0.2 # Rainbow (3) Prioritized Experience Replay
    beta = 0.6 # Rainbow (3) Prioritized Experience Replay
    try:
        # ff_test(env_manager)
        # pre_train(env_manager, main_agent, encoder_lr = 1e-3, vnf_s_lr = 1e-3, vnf_p_lr = 1e-3, tot_episode_num = 5_000, gamma = GAMMA)
        # update_target(main_agent, target_agent)
        live_train(env_manager, main_agent, target_agent, encoder_lr = 1e-4, vnf_s_lr = 1e-4, vnf_p_lr = 1e-4, tot_episode_num = 2_000, gamma = GAMMA, alpha=alpha, beta=beta)
    finally:
        env_manager.delete_all()
