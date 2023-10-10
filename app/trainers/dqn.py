import os
import json
import torch
import numpy as np
from copy import deepcopy

from app.agents.dqn import DQNAgent, DQNAgentInfo
from app.dl.models.encoder import EncoderInfo
from app.dl.models.core import StateEncoderInfo
from app.dl.models.dqn import DQNValueInfo
from app.agents.random import RandomAgent
from app.utils import utils

from app.agents.dqn import DQNAgent
from app.agents.random import RandomAgent
from app.trainers.env import Environment
from app.k8s.envManager import EnvManager
from app.constants import *
from app.trainers.memory import ReplayMemory, Data
from app.trainers.debug import Debugger
from app.types import Episode

def live_train(env_manager: EnvManager, dqn_agent: DQNAgent, encoder_lr: float, vnf_s_lr: float, vnf_p_lr: float, tot_episode_num: int, gamma: float):
    # 1. setup env
    
    env = env_manager.create_env(id="dqn-live-train")
    random_agent = RandomAgent()


    # 2. setup optimizers and loss_fn
    encoder_optimizer = torch.optim.Adam(dqn_agent.encoder.parameters(), lr=encoder_lr)
    vnf_s_optimizer = torch.optim.Adam(dqn_agent.vnf_s_value.parameters(), lr=vnf_s_lr)
    vnf_p_optimizer = torch.optim.Adam(dqn_agent.vnf_p_value.parameters(), lr=vnf_p_lr)
    
    loss_fn = torch.nn.HuberLoss()

    # 3. setup replay memory
    batch_size = 8
    seq_len = 5
    memory = ReplayMemory(batch_size, seq_len, 10_000)

    # 4. set debugging pockets
    debugger = Debugger()

    # 5. run live training
    # - exploration을 조금씩 줄여나가면서 업데이트 진행
    # - run dqn three time
    # 에피소드를 전체 한 번에 받고,
    # 해당 episode를 활용해서 update하기
    epsilon = 0.5
    for episode_num in range(1, tot_episode_num + 1):
        state, info, done = env.reset()
        explore_rate = epsilon * (1 - episode_num / tot_episode_num)
        ini_state = state
        ini_info = info
        for step_num in range(1, 101):
            if np.random.uniform() < explore_rate:
                # random
                action = random_agent.inference(state)
            else:
                # greedy
                action = dqn_agent.inference(state)
            next_state, next_info, done = env.step(action)
            memory.append(Data(
                episode_num=episode_num, 
                step_num=step_num,
                state=state,
                info=info,
                action=action,
                next_state=next_state,
                next_info=next_info,
                done=done,
            ))
            if done:
                break

            state = next_state

            batch = memory.sample()
            if len(batch) < batch_size:
                continue
            state_seq_batch = [[sample.state if sample != None else None for sample in seq] for seq in batch]
            vnf_s_action_batch = torch.tensor([seq[-1].action.vnfId for seq in batch])
            vnf_p_action_batch = torch.tensor([seq[-1].action.srvId for seq in batch])
            reward_seq_batch = [[utils.calc_reward(sample.info, sample.next_info) if sample != None else 0 for sample in seq] for seq in batch]
            reward_batch = torch.tensor([utils.calc_reward(seq[-1].info, seq[-1].next_info) for seq in batch])
            next_state_batch = [[seq[-1].next_state] for seq in batch]

            dqn_agent.encoder.train()

            rack_x, srv_x, sfc_x, vnf_x, core_x = dqn_agent.encoder(state_seq_batch)

            next_rack_x, next_srv_x, next_sfc_x, next_vnf_x, next_core_x = dqn_agent.encoder(next_state_batch)

            next_srv_x = next_srv_x.detach()
            next_vnf_x = next_vnf_x.detach()
            next_core_x = next_core_x.detach()

            dqn_agent.vnf_s_value.train()
            
            vnf_s_value = dqn_agent.vnf_s_value(
                core_x.unsqueeze(1).repeat(1, vnf_x.shape[1], 1),
                vnf_x,
                vnf_x.clone(),
            )

            dqn_agent.vnf_s_value.eval()

            next_vnf_s_value = dqn_agent.vnf_s_value(
                next_core_x.unsqueeze(1).repeat(1, next_vnf_x.shape[1], 1),
                next_vnf_x,
                next_vnf_x.clone(),
            )
            vnf_s_q = vnf_s_value.gather(1, vnf_s_action_batch.unsqueeze(1))
            vnf_s_expected_q = (reward_batch + gamma * next_vnf_s_value.max(1)[0].detach()).unsqueeze(1)

            dqn_agent.vnf_p_value.train()
            
            vnf_p_value = dqn_agent.vnf_p_value(
                vnf_s_value.unsqueeze(1).repeat(1, srv_x.shape[1], 1),
                srv_x,
                srv_x.clone(),
            )

            dqn_agent.vnf_p_value.eval()

            next_vnf_p_value = dqn_agent.vnf_p_value(
                next_vnf_s_value.unsqueeze(1).repeat(1, next_srv_x.shape[1], 1),
                next_srv_x,
                next_srv_x.clone(),
            )

            vnf_p_q = vnf_p_value.gather(1, vnf_p_action_batch.unsqueeze(1))
            vnf_p_expected_q = (reward_batch + gamma * next_vnf_p_value.max(1)[0].detach()).unsqueeze(1)

            dqn_agent.vnf_s_value.train()
            dqn_agent.vnf_p_value.train()

            vnf_s_loss = loss_fn(vnf_s_q, vnf_s_expected_q)
            vnf_p_loss = loss_fn(vnf_p_q, vnf_p_expected_q)
            vnf_s_p_loss = vnf_s_loss + vnf_p_loss
            encoder_optimizer.zero_grad()
            vnf_s_optimizer.zero_grad()
            vnf_p_optimizer.zero_grad()
            vnf_s_p_loss.backward()
            encoder_optimizer.step()
            vnf_s_optimizer.step()
            vnf_p_optimizer.step()

        fin_state = state
        fin_info = info

        debugger.add_episode(ini_state, ini_info, fin_state, fin_info, explore_rate, step_num)
        debugger.print(last_n=100)



def pre_train(dqn_agent: DQNAgent, encoder_lr: float, vnf_s_lr: float, vnf_p_lr: float, tot_episode_num: int, gamma: float):
    # 1. setup env

    # 2. setup optimizers and loss_fn
    encoder_optimizer = torch.optim.Adam(dqn_agent.encoder.parameters(), lr=encoder_lr)
    vnf_s_optimizer = torch.optim.Adam(dqn_agent.vnf_s_value.parameters(), lr=vnf_s_lr)
    vnf_p_optimizer = torch.optim.Adam(dqn_agent.vnf_p_value.parameters(), lr=vnf_p_lr)
    
    loss_fn = torch.nn.HuberLoss()

    # 3. setup replay memory
    batch_size = 8
    seq_len = 5
    memory = ReplayMemory(batch_size, seq_len, 100_000)

    pre_data_path = "./data/episode/random"
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
            actionList = list(map(lambda step: step.action, episode.steps))
            infoList = list(map(lambda step: step.info, episode.steps))
            
            for i in range(len(stateList) - 1):
                done = not infoList[i+1].success
                if done: break
                memory.append(Data(
                    episode_num=episode_num, 
                    step_num=i,
                    state=stateList[i],
                    info=infoList[i],
                    action=actionList[i],
                    next_state=stateList[i+1],
                    next_info=infoList[i+1],
                    done=done,
                ))

    # 4. run live training
    # - exploration을 조금씩 줄여나가면서 업데이트 진행
    # - run dqn three time
    # 에피소드를 전체 한 번에 받고,
    # 해당 episode를 활용해서 update하기
    for episode_num in range(1, tot_episode_num + 1):
        batch = memory.sample()
        if len(batch) < batch_size:
            continue
        state_seq_batch = [[sample.state if sample != None else None for sample in seq] for seq in batch]
        vnf_s_action_batch = torch.tensor([seq[-1].action["vnfId"] for seq in batch])
        vnf_p_action_batch = torch.tensor([seq[-1].action["srvId"] for seq in batch])
        reward_seq_batch = [[utils.calc_reward(sample.info, sample.next_info) if sample != None else 0 for sample in seq] for seq in batch]
        reward_batch = torch.tensor([utils.calc_reward(seq[-1].info, seq[-1].next_info) for seq in batch])
        next_state_batch = [[seq[-1].next_state] for seq in batch]

        dqn_agent.encoder.train()

        rack_x, srv_x, sfc_x, vnf_x, core_x = dqn_agent.encoder(state_seq_batch)

        dqn_agent.encoder.eval()

        next_rack_x, next_srv_x, next_sfc_x, next_vnf_x, next_core_x = dqn_agent.encoder(next_state_batch)

        next_srv_x = next_srv_x.detach()
        next_vnf_x = next_vnf_x.detach()
        next_core_x = next_core_x.detach()

        dqn_agent.vnf_s_value.train()
        
        vnf_s_value = dqn_agent.vnf_s_value(
            core_x.unsqueeze(1).repeat(1, vnf_x.shape[1], 1),
            vnf_x,
            vnf_x.clone(),
        )

        dqn_agent.vnf_s_value.eval()

        next_vnf_s_value = dqn_agent.vnf_s_value(
            next_core_x.unsqueeze(1).repeat(1, next_vnf_x.shape[1], 1),
            next_vnf_x,
            next_vnf_x.clone(),
        )
        vnf_s_q = vnf_s_value.gather(1, vnf_s_action_batch.unsqueeze(1))
        vnf_s_expected_q = (reward_batch + gamma * next_vnf_s_value.max(1)[0].detach()).unsqueeze(1)

        dqn_agent.vnf_p_value.train()
        
        vnf_p_value = dqn_agent.vnf_p_value(
            vnf_s_value.unsqueeze(1).repeat(1, srv_x.shape[1], 1),
            srv_x,
            srv_x.clone(),
        )

        dqn_agent.vnf_p_value.eval()

        next_vnf_p_value = dqn_agent.vnf_p_value(
            next_vnf_s_value.unsqueeze(1).repeat(1, next_srv_x.shape[1], 1),
            next_srv_x,
            next_srv_x.clone(),
        )

        vnf_p_q = vnf_p_value.gather(1, vnf_p_action_batch.unsqueeze(1))
        vnf_p_expected_q = (reward_batch + gamma * next_vnf_p_value.max(1)[0].detach()).unsqueeze(1)

        dqn_agent.vnf_s_value.train()
        dqn_agent.vnf_p_value.train()

        vnf_s_loss = loss_fn(vnf_s_q, vnf_s_expected_q)
        vnf_p_loss = loss_fn(vnf_p_q, vnf_p_expected_q)
        vnf_s_p_loss = vnf_s_loss + vnf_p_loss
        encoder_optimizer.zero_grad()
        vnf_s_optimizer.zero_grad()
        vnf_p_optimizer.zero_grad()
        vnf_s_p_loss.backward()
        encoder_optimizer.step()
        vnf_s_optimizer.step()
        vnf_p_optimizer.step()
    print(f"[PreTrain] Episode #{episode_num} done")


def test():
    pass



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
    dqn_agent = DQNAgent(DQNAgentInfo(
        encoder_info=stateEncoderInfo,
        vnf_s_value_info=DQNValueInfo(
            query_size=8,
            key_size=8,
            value_size=8,
            hidden_sizes=[8, 8],
            num_heads=[4, 4],
            dropout=0.3,
        ),
        vnf_p_value_info=DQNValueInfo(
            query_size=MAX_VNF_NUM,
            key_size=4,
            value_size=4,
            hidden_sizes=[8, 8],
            num_heads=[4, 4],
            dropout=0.3,
        ),
    ))
    pre_train(dqn_agent, encoder_lr = 0.001, vnf_s_lr = 0.001, vnf_p_lr = 0.001, tot_episode_num = 1000, gamma = 0.99)
    env_manager = EnvManager()
    try:
        live_train(env_manager, dqn_agent, encoder_lr = 0.001, vnf_s_lr = 0.001, vnf_p_lr = 0.001, tot_episode_num = 1000, gamma = 0.99)
    finally:
        env_manager.delete_all()


