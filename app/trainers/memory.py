import gc
import time
from collections import deque
from typing import List, Optional, Deque
from dataclasses import dataclass

import numpy as np
import torch

from app.types import State, Info, Action
from app.trainers.env import MultiprocessEnvironment
from app.agents.ppo import PPOAgent

from app.utils import utils

@dataclass
class Data:
    episode_num: int
    step_num: int
    state: State
    info: Info
    action: Action
    done: bool
    next_state: State
    next_info: Info

class ReplayMemory:
    def __init__(self, batch_size, seq_len, max_memory_len=10_000):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_memory_len = max_memory_len
        self.buffer: Deque[Data] = deque(maxlen=max_memory_len)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def sample(self) -> List[List[Optional[Data]]]:
        if len(self.buffer) < self.batch_size:
            return []
        sample_idxs = np.random.choice(np.arange(0, len(self.buffer)), self.batch_size, replace=False)
        batch = []
        # 기본적으로 선택된 sample idx에서 뒤로 seq만큼의 data를 추가 추출해서 seq를 만든다.
        # 하지만, 길이가 부족한 경우에는 None로 채운다.
        for sample_idx in sample_idxs:
            sample = []
            prev = self.buffer[sample_idx]
            sample.append(prev)
            for i in range(1, self.seq_len):
                cur_idx = sample_idx - i
                if cur_idx < 0:
                    sample.append(None)
                    continue
                cur = self.buffer[cur_idx]
                if cur.episode_num != prev.episode_num:
                    sample.append(None)
                    continue
                sample.append(cur)
                prev = cur
            sample.reverse()
            batch.append(sample)
        return batch
    
    def append(self, data: Data) -> None:
        self.buffer.append(data)

@dataclass
class EpisodeData:
    episode_num: int
    episode_length: int
    states: List[State]
    info: List[Info]
    action: List[Action]
    done: List[bool]
    next_state: List[State]
    next_info: List[Info]

    

class EpisodeMemory:
    def __init__(
            self, 
            mp_env: MultiprocessEnvironment,
            n_workers: int, batch_size: int, seq_len: int,
            gamma: float, tau: float, 
            episode_num: int, max_episode_len: int,
        ):
        self.mp_env = mp_env
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.gamma = gamma
        self.tau = tau
        self.episode_num = episode_num
        self.max_episode_len = max_episode_len

        self.discounts = torch.logspace(0, max_episode_len+1, steps=max_episode_len+1, base=gamma, dtype=torch.float64)
        self.tau_discounts = torch.logspace(0, max_episode_len+1, steps=max_episode_len+1, base=(gamma*tau), dtype=torch.float64)

        self.reset()


    def reset(self):
        self.cur_episode_idxs = torch.arange((self.n_workers), dtype=torch.int32) # torch.tensor (n_workers,)

        self.infos = [[None for _ in range(self.max_episode_len)] for _ in range(self.episode_num)]         # List[List[Info]] (episode_num, max_episode_len)
        self.states = [[None for _ in range(self.max_episode_len)] for _ in range(self.episode_num)]        # List[List[State]] (episode_num, max_episode_len)
        self.actions = [[None for _ in range(self.max_episode_len)] for _ in range(self.episode_num)]       # List[List[Action]] (episode_num, max_episode_len)
        self.next_infos = [[None for _ in range(self.max_episode_len)] for _ in range(self.episode_num)]    # List[List[Info]] (episode_num, max_episode_len)
        self.next_states = [[None for _ in range(self.max_episode_len)] for _ in range(self.episode_num)]   # List[List[State]] (episode_num, max_episode_len) 

        self.vnf_s_logpas = torch.empty((self.episode_num, self.max_episode_len))   # torch.tensor (episode_num, max_episode_len)
        self.vnf_p_logpas = torch.empty((self.episode_num, self.max_episode_len))   # torch.tensor (episode_num, max_episode_len)
        self.vnf_returns = torch.empty((self.episode_num, self.max_episode_len))    # torch.tensor (episode_num, max_episode_len)
        self.vnf_values = torch.empty((self.episode_num, self.max_episode_len))     # torch.tensor (episode_num, max_episode_len)
        self.vnf_gaes = torch.empty((self.episode_num, self.max_episode_len))       # torch.tensor (episode_num, max_episode_len)

        self.episode_lens = torch.zeros((self.episode_num), dtype=torch.int32)              # torch.tensor (episode_num,)
        self.episode_rewards = torch.zeros((self.episode_num), dtype=torch.float64)         # torch.tensor (episode_num,)
        self.episode_seconds = torch.zeros((self.episode_num), dtype=torch.float64)         # torch.tensor (episode_num,)
        self.episode_explorations = torch.zeros((self.episode_num), dtype=torch.float64)    # torch.tensor (episode_num,)

        gc.collect()

    def make_prev_seq_state(self, episode_idx, step_idx):
        prev_seq_state = []
        if step_idx < self.seq_len:
            prev_seq_state = [None for _ in range(self.seq_len - step_idx - 1)]
        prev_seq_state += self.states[episode_idx][max(0, step_idx - self.seq_len):step_idx]
        return prev_seq_state


    # fill memory with n_workers.
    def fill(self, agent: PPOAgent):
        workers_explorations = torch.zeros((self.n_workers, self.max_episode_len, 2), dtype=torch.float32)
        workers_steps = torch.zeros((self.n_workers), dtype=torch.int32)
        workers_seconds = torch.tensor([time.time(), ] * self.n_workers, dtype=torch.float64)
        workers_rewards = torch.zeros((self.n_workers, self.max_episode_len), dtype=torch.float32)

        agent.encoder.eval()
        agent.vnf_value.eval()
        agent.vnf_s_policy.eval()
        agent.vnf_p_policy.eval()

        states, infos, dones = self.mp_env.reset()

        # TODO: Sequence 길이 적용
        # States 한칸씩 미는 기능 추가하고, 처음 시작하면 다시 넣기. 
        while len(self.episode_lens[self.episode_lens > 0]) < self.max_episode_len:
            with torch.no_grad():
                rack_x, srv_x, sfc_x, vnf_x, core_x = agent.encoder([self.make_prev_seq_state(e_idx, s_idx) + [state] for e_idx, s_idx, state in zip(self.cur_episode_idxs, workers_steps, states)])
                vnf_s_out = agent.vnf_s_policy(
                    core_x.unsqueeze(1).repeat(1, vnf_x.shape[1], 1),
                    vnf_x,
                    vnf_x.clone(),
                )
                vnf_s_actions, vnf_s_logpas, vnf_s_is_exploratory = utils.get_info_from_logits(vnf_s_out)
                vnf_p_out = agent.vnf_p_policy(
                    vnf_s_out.unsqueeze(1).repeat(1, srv_x.shape[1], 1),
                    srv_x,
                    srv_x.clone(),
                )
                vnf_p_actions, vnf_p_logpas, vnf_p_is_exploratory = utils.get_info_from_logits(vnf_p_out)
            actions = [Action(vnfId=vnf_s_actions[i].item(), srvId=vnf_p_actions[i].item()) for i in range(self.n_workers)]
            next_states, next_infos, next_dones = self.mp_env.step(actions)
            rewards = torch.tensor([utils.calc_reward(info, next_info) for info, next_info in zip(infos, next_infos)], dtype=torch.float32)
            next_dones = torch.tensor(next_dones, dtype=torch.bool)

            for cur_episode_idx in self.cur_episode_idxs:
                for worker_idx, cur_step_idx in enumerate(workers_steps):
                    self.infos[cur_episode_idx][cur_step_idx] = infos[worker_idx]
                    self.states[cur_episode_idx][cur_step_idx] = states[worker_idx] 
                    self.actions[cur_episode_idx][cur_step_idx] = actions[worker_idx]
                    self.next_infos[cur_episode_idx][cur_step_idx] = next_infos[worker_idx]
                    self.next_states[cur_episode_idx][cur_step_idx] = next_states[worker_idx]

            self.vnf_s_logpas[self.cur_episode_idxs] = vnf_s_logpas
            self.vnf_p_logpas[self.cur_episode_idxs] = vnf_p_logpas
            
            workers_explorations[torch.arange(self.n_workers), workers_steps] = torch.stack([vnf_s_is_exploratory.to(torch.float32), vnf_p_is_exploratory.to(torch.float32)], dim=1)
            workers_rewards[torch.arnage(self.n_workers), workers_steps] = rewards

            if next_dones.sum() > 0:
                idx_done = torch.where(next_dones)[0]
                next_values = torch.zeros((self.n_workers))
                with torch.no_grad():
                    rack_x, srv_x, sfc_x, vnf_x, core_x = agent.encoder([self.make_prev_seq_state(e_idx, s_idx + 1) + [next_state] for e_idx, s_idx, next_state in zip(self.cur_episode_idxs, workers_steps, next_states)])
                    values = agent.vnf_value(core_x, core_x, core_x)
                    next_values[idx_done] = values[idx_done]

            states = next_states
            infos = next_infos
            dones = next_dones

            workers_steps += 1

            if next_dones.sum() > 0:
                new_states, new_infos, new_dones = self.mp_env.reset(idx_done)
                for new_s_idx, s_idx in enumerate(idx_done):
                    states[s_idx] = new_states[new_s_idx]
                    infos[s_idx] = new_infos[new_s_idx]
                    dones[s_idx] = new_dones[new_s_idx]
                for w_idx in range(self.n_workers):
                    if w_idx not in idx_done: continue
                    e_idx =self.cur_episode_idxs[w_idx]
                    T = workers_steps[w_idx]
                    self.episode_steps[e_idx] = T
                    self.episode_rewards[e_idx] = workers_rewards[w_idx, : T].sum()
                    self.episode_explorations[e_idx] = workers_explorations[w_idx, : T].mean()
                    self.episode_seconds[e_idx] = time.time() - workers_seconds[w_idx]

                    ep_rewards = torch.concat([workers_rewards[w_idx, :T], next_values[w_idx].unsqueeze(0)], dim=0)
                    ep_discounts = self.discounts[:T+1]
                    ep_returns = torch.Tensor([(ep_discounts[:T+1-t] * ep_rewards[t:]).sum() for t in range(T)])
                    self.returns[e_idx, :T] = ep_returns

                    ep_states = self.states[e_idx, :T]

                    with torch.no_grad():
                        rack_x, srv_x, sfc_x, vnf_x, core_x = agent.encoder([self.make_prev_seq_state(e_idx, s_idx + 1) + [ep_state] for e_idx, s_idx, ep_state in zip(self.cur_episode_idxs, workers_steps, ep_states)])
                        ep_values = agent.vnf_value(core_x, core_x, core_x)
                        ep_values = torch.cat([ep_values, next_values[w_idx].unsqueeze(0)], dim=0)
                    
                    ep_deltas = ep_rewards[:-1] + self.gamma * ep_values[1:] - ep_values[:-1]
                    ep_gaes = torch.Tensor([(ep_discounts[:T-t] * ep_deltas[t:]).sum() for t in range(T)])

                    self.gaes[e_idx, :T] = ep_gaes

                    workers_explorations[w_idx, :] = False
                    workers_rewards[w_idx, :] = 0
                    workers_steps[w_idx] = 0
                    workers_seconds[w_idx] = time.time()

                    new_ep_id = max(self.cur_episode_idxs) + 1
                    if new_ep_id >= self.memory_max_episode_num:
                        break
                    self.cur_episode_idxs[w_idx] = new_ep_id

        # 마무리로 각 step 길이 만큼만 들어가도록 None 데이터 자르기
        ep_idxs = self.episode_steps > 0
        ep_steps = self.episode_steps[ep_idxs]
       
        self.states = [row[:t] for row, t in zip(self.states, ep_steps)]
        self.actions = [row[:t] for row, t in zip(self.actions, ep_steps)]
        self.values = torch.concat([row[:t] for row, t in zip(self.values, ep_steps)])
        self.returns = torch.concat([row[:t] for row, t in zip(self.returns, ep_steps)])
        self.gaes = torch.concat([row[:t] for row, t in zip(self.gaes[ep_idxs], ep_steps)])
        self.logpas = torch.concat([row[:t] for row, t in zip(self.logpas[ep_idxs], ep_steps)])

        ep_rewards = self.episode_rewards[ep_idxs]
        ep_explorations = self.episode_explorations[ep_idxs]
        ep_seconds = self.episode_seconds[ep_idxs]

        return ep_steps, ep_rewards, ep_explorations, ep_seconds

        

    def sample(self):
        if all:
            return self.states, self.actions, self.returns, self.gaes, self.vnf_s_logpas, self.vnf_p_logpas, self.values
        mem_size = len(self)
        batch_idxs = np.random.choice(mem_size, self.batch_size, replace=False)
        states = [self.states[batch_idx] for batch_idx in batch_idxs]
        actions = [self.actions[batch_idx] for batch_idx in batch_idxs]
        
        values = self.values[batch_idxs]
        returns = self.returns[batch_idxs]
        gaes = self.gaes[batch_idxs]
        gaes = (gaes - gaes.mean()) / gaes.std() + 1e-8
        vnf_s_logpas = self.vnf_s_logpas[batch_idxs]
        vnf_p_logpas = self.vnf_p_logpas[batch_idxs]

        return states, actions, returns, gaes, vnf_s_logpas, vnf_p_logpas, values

    def get_debugger(self): pass


    def __len__(self):
        return len(self.actions)