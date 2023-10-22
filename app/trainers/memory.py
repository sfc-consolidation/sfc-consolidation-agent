import os
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
from app.utils.segment_tree import MinSegmentTree, SumSegmentTree
from app.constants import *

@dataclass
class Data:
    episode_num: int
    step_num: int
    state: State
    action: Action
    reward: float
    done: bool
    next_state: State

class ReplayMemory:
    def __init__(self, batch_size, seq_len, max_memory_len=10_000, n_step=1, gamma=0.9):
        self.gamma = gamma
        self.n_step = n_step
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_memory_len = max_memory_len
        self.buffer: Deque[Data] = deque(maxlen=max_memory_len)
        self.one_step_buffer: Deque[Data] = deque(maxlen=n_step) # Rainbow (7) N-Step Buffer
    
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
        if len(self.one_step_buffer) > 0:
            first_data = self.one_step_buffer[0]
            if first_data.episode_num != data.episode_num:
                self.one_step_buffer.clear()
                self.one_step_buffer.append(data)
            else:
                self.one_step_buffer.append(data)
        else:
            self.one_step_buffer.append(data)
        
        if len(self.one_step_buffer) == self.n_step:
            n_step_data = self._get_n_step_data()
            self.buffer.append(n_step_data)
    
    def _get_n_step_data(self) -> Data:
        n_step_state = self.one_step_buffer[0].state
        n_step_action = self.one_step_buffer[0].action

        n_step_reward = self.one_step_buffer[-1].reward
        n_step_next_state = self.one_step_buffer[-1].next_state
        n_step_done = self.one_step_buffer[-1].done
        
        for data in reversed(list(self.one_step_buffer)[:-1]):
            r = data.reward
            n_s = data.next_state
            d = data.done

            n_step_reward = r + self.gamma * n_step_reward * (1 - d)
            n_step_next_state = n_s if d else n_step_next_state
            n_step_done = d
        
        n_step_data = Data(
            episode_num=self.one_step_buffer[0].episode_num,
            step_num=self.one_step_buffer[0].step_num,
            state=n_step_state,
            action=n_step_action,
            reward=n_step_reward,
            done=n_step_done,
            next_state=n_step_next_state,
        )
        return n_step_data

# Rainbow (3) Prioritized Replay Memory
class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, alpha, batch_size, seq_len, max_memory_len=10_000, n_step=1, gamma=0.9):
        super().__init__(batch_size, seq_len, max_memory_len, n_step, gamma)
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha

        tree_capacity = 1
        while tree_capacity < max_memory_len:
            tree_capacity *= 2
        
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def append(self, data: Data):
        super().append(data)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_memory_len

    def sample(self, beta):
        assert len(self) >= self.batch_size
        assert beta > 0

        sample_idxs = self._sample_proportional()
        weights = np.array([self._calculate_weight(i, beta) for i in sample_idxs])

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
        return batch, weights, sample_idxs


    def update_priorities(self, idxs, priorities):
        assert len(idxs) == len(priorities)
        for idx, priority in zip(idxs, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, max(priorities))

    def _sample_proportional(self):
        idxs = []
        p_total = self.sum_tree.sum(0, len(self.buffer) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            idxs.append(idx)
        return idxs

    def _calculate_weight(self, idx, beta):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self.buffer)) ** (-beta)
        weight = weight / max_weight

        return weight

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
        # for multiprocessing
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['OMP_NUM_THREADS'] = '1'

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
        self.returns = torch.empty((self.episode_num, self.max_episode_len))    # torch.tensor (episode_num, max_episode_len)
        self.values = torch.empty((self.episode_num, self.max_episode_len))     # torch.tensor (episode_num, max_episode_len)
        self.gaes = torch.empty((self.episode_num, self.max_episode_len))       # torch.tensor (episode_num, max_episode_len)

        self.episode_lens = torch.zeros((self.episode_num), dtype=torch.int32)              # torch.tensor (episode_num,)
        self.episode_rewards = torch.zeros((self.episode_num), dtype=torch.float64)         # torch.tensor (episode_num,)
        self.episode_seconds = torch.zeros((self.episode_num), dtype=torch.float64)         # torch.tensor (episode_num,)
        self.episode_explorations = torch.zeros((self.episode_num), dtype=torch.float64)    # torch.tensor (episode_num,)

        gc.collect()

    def make_prev_seq_state(self, episode_idx, step_idx):
        prev_seq_state = [None for _ in range(max(0, self.seq_len - step_idx - 1))]
        prev_seq_state += self.states[episode_idx][max(0, step_idx - self.seq_len + 1):step_idx]
        return prev_seq_state


    # fill memory with n_workers.
    async def fill(self, agent: PPOAgent, resetArg):
        workers_explorations = torch.zeros((self.n_workers, self.max_episode_len, 2), dtype=torch.float32)
        workers_steps = torch.zeros((self.n_workers), dtype=torch.int32)
        workers_seconds = torch.tensor([time.time(), ] * self.n_workers, dtype=torch.float64)
        workers_rewards = torch.zeros((self.n_workers, self.max_episode_len), dtype=torch.float32)

        agent.encoder.eval()
        agent.vnf_value.eval()
        agent.vnf_s_policy.eval()
        agent.vnf_p_policy.eval()

        states, infos, dones = await self.mp_env.reset(resetArg=resetArg)
        
        while len(self.episode_lens[self.episode_lens > 0]) < self.episode_num / 2:
            with torch.no_grad():
                mp_seq_state = [self.make_prev_seq_state(e_idx, s_idx) + [state] for e_idx, s_idx, state in zip(self.cur_episode_idxs, workers_steps, states)]
                action_mask = utils.get_possible_action_mask(mp_seq_state).to(TORCH_DEVICE)
                
                rack_x, srv_x, sfc_x, vnf_x, core_x = agent.encoder(mp_seq_state)
                vnf_s_out = agent.vnf_s_policy(
                    core_x.unsqueeze(1).repeat(1, vnf_x.shape[1], 1),
                    vnf_x,
                    vnf_x.clone(),
                )
                vnf_s_mask = action_mask.sum(dim=2) == 0
                vnf_s_out = vnf_s_out.masked_fill(vnf_s_mask, 0)

                vnf_idxs, vnf_s_logpas, vnf_s_is_exploratory = utils.get_info_from_logits(vnf_s_out)
                
                vnf_p_out = agent.vnf_p_policy(
                    vnf_s_out.unsqueeze(1).repeat(1, srv_x.shape[1], 1),
                    srv_x,
                    srv_x.clone(),
                )
                vnf_p_mask = action_mask[torch.arange(vnf_idxs.shape[0]), vnf_idxs, :] == 0
                vnf_p_out = vnf_p_out.masked_fill(vnf_p_mask, 0)

                srv_idxs, vnf_p_logpas, vnf_p_is_exploratory = utils.get_info_from_logits(vnf_p_out)
            
            actions = [Action(vnfId=vnf_idxs[i].item(), srvId=srv_idxs[i].item()) for i in range(self.n_workers)]
            
            next_states, next_infos, is_fails = await self.mp_env.step(actions)

            rewards = torch.tensor([utils.calc_reward(info, next_info) for info, next_info in zip(infos, next_infos)], dtype=torch.float32)
            is_fails = torch.tensor(is_fails, dtype=torch.bool)

            for cur_episode_idx in self.cur_episode_idxs:
                for worker_idx, cur_step_idx in enumerate(workers_steps):
                    self.infos[cur_episode_idx][cur_step_idx] = infos[worker_idx]
                    self.states[cur_episode_idx][cur_step_idx] = states[worker_idx] 
                    self.actions[cur_episode_idx][cur_step_idx] = actions[worker_idx]
                    self.next_infos[cur_episode_idx][cur_step_idx] = next_infos[worker_idx]
                    self.next_states[cur_episode_idx][cur_step_idx] = next_states[worker_idx]

            self.vnf_s_logpas[self.cur_episode_idxs, workers_steps] = vnf_s_logpas.cpu()
            self.vnf_p_logpas[self.cur_episode_idxs, workers_steps] = vnf_p_logpas.cpu()
            
            workers_explorations[torch.arange(self.n_workers), workers_steps] = torch.stack([vnf_s_is_exploratory.to(torch.float32).cpu(), vnf_p_is_exploratory.to(torch.float32).cpu()], dim=1)
            workers_rewards[torch.arange(self.n_workers), workers_steps] = rewards

            # calculate V(s_t+1)
            # at least one worker is done
            if is_fails.sum() > 0:
                failed_worker_idx = torch.where(is_fails)[0]
                next_values = torch.zeros((self.n_workers))
                with torch.no_grad():
                    mp_seq_next_state = [self.make_prev_seq_state(e_idx, s_idx) + [state] for e_idx, s_idx, state in zip(self.cur_episode_idxs, workers_steps, states)]
                    rack_x, srv_x, sfc_x, vnf_x, core_x = agent.encoder(mp_seq_next_state)
                    values = agent.vnf_value(core_x).cpu().squeeze(1)
                    next_values[failed_worker_idx] = values[failed_worker_idx]

            # all workers are step forward
            states = next_states
            infos = next_infos

            workers_steps += 1

            # if done, then reset
            if is_fails.sum() > 0:
                new_states, new_infos, new_dones = await self.mp_env.reset(failed_worker_idx, resetArg=resetArg)
                for new_s_idx, s_idx in enumerate(failed_worker_idx):
                    states[s_idx] = new_states[new_s_idx]
                    infos[s_idx] = new_infos[new_s_idx]
                for w_idx in range(self.n_workers):
                    if w_idx not in failed_worker_idx: continue
                    e_idx = self.cur_episode_idxs[w_idx]
                    if workers_steps[w_idx] < 2:
                        workers_explorations[w_idx, :] = False
                        workers_rewards[w_idx, :] = 0
                        workers_steps[w_idx] = 0
                        workers_seconds[w_idx] = time.time()
                        continue
                    T = workers_steps[w_idx]
                    self.episode_lens[e_idx] = T - 1
                    self.episode_rewards[e_idx] = workers_rewards[w_idx, :T].sum()
                    self.episode_explorations[e_idx] = workers_explorations[w_idx, : T].mean()
                    self.episode_seconds[e_idx] = time.time() - workers_seconds[w_idx]

                    ep_rewards = torch.concat([workers_rewards[w_idx, :T], next_values[w_idx].unsqueeze(0)], dim=0) # torch.tensor (T+1)
                    ep_discounts = self.discounts[:T+1]
                    ep_returns = torch.Tensor([(ep_discounts[:T+1-t] * ep_rewards[t:]).sum() for t in range(T)])
                    self.returns[e_idx, :T] = ep_returns

                    ep_states = self.states[e_idx][:T]

                    with torch.no_grad():
                        full_ep_seq_state = [self.make_prev_seq_state(e_idx, s_idx) + [ep_state] for s_idx, ep_state in enumerate(ep_states)]
                        rack_x, srv_x, sfc_x, vnf_x, core_x = agent.encoder(full_ep_seq_state)
                        ep_values = agent.vnf_value(core_x).cpu().squeeze(1)
                        ep_values = torch.cat([ep_values, next_values[w_idx].unsqueeze(0)], dim=0)
                    
                    ep_deltas = ep_rewards[:-1] + self.gamma * ep_values[1:] - ep_values[:-1]
                    ep_gaes = torch.Tensor([(ep_discounts[:T-t] * ep_deltas[t:]).sum() for t in range(T)])

                    self.gaes[e_idx, :T] = ep_gaes

                    workers_explorations[w_idx, :] = False
                    workers_rewards[w_idx, :] = 0
                    workers_steps[w_idx] = 0
                    workers_seconds[w_idx] = time.time()

                    new_ep_id = max(self.cur_episode_idxs) + 1
                    if new_ep_id >= self.episode_num:
                        break
                    self.cur_episode_idxs[w_idx] = new_ep_id

        # 마무리로 각 step 길이 만큼만 들어가도록 None 데이터 자르기
        ep_idxs = self.episode_lens > 0
        ep_steps = self.episode_lens[ep_idxs]
       
        self.infos = [row[:t] for row, t in zip(self.infos, ep_steps)]
        self.states = [row[:t] for row, t in zip(self.states, ep_steps)]
        self.actions = [row[:t] for row, t in zip(self.actions, ep_steps)]
        self.next_infos = [row[:t] for row, t in zip(self.next_infos, ep_steps)]
        self.next_states = [row[:t] for row, t in zip(self.next_states, ep_steps)]

        self.values = torch.concat([row[:t] for row, t in zip(self.values, ep_steps)])
        self.returns = torch.concat([row[:t] for row, t in zip(self.returns, ep_steps)])
        self.gaes = torch.concat([row[:t] for row, t in zip(self.gaes[ep_idxs], ep_steps)])
        self.vnf_s_logpas = torch.concat([row[:t] for row, t in zip(self.vnf_s_logpas[ep_idxs], ep_steps)])
        self.vnf_p_logpas = torch.concat([row[:t] for row, t in zip(self.vnf_p_logpas[ep_idxs], ep_steps)])

        ep_rewards = self.episode_rewards[ep_idxs]
        ep_explorations = self.episode_explorations[ep_idxs]
        ep_seconds = self.episode_seconds[ep_idxs]

        return ep_steps, ep_rewards, ep_explorations, ep_seconds

        
    # TODO: seq length만큼 끊어주기
    def sample(self, all=False):
        if all:
            return self.states, self.actions, self.returns, self.gaes, self.vnf_s_logpas, self.vnf_p_logpas, self.values
        mem_size = len(self)
        batch_idxs = np.random.choice(mem_size, self.batch_size, replace=False)
        seq_last_idxs = []
        for batch_idx in batch_idxs:
            seq_last_idx = np.random.choice(len(self.states[batch_idx]), 1, replace=False)[0]
            seq_last_idxs.append(seq_last_idx)
        states = [self.states[batch_idx][max(0, seq_last_idx+1 - self.seq_len):seq_last_idx+1] for batch_idx, seq_last_idx in zip(batch_idxs, seq_last_idxs)]
        for idx, seq_state in enumerate(states):
            if len(seq_state) < self.seq_len:
                seq_state = [None for _ in range(self.seq_len - len(seq_state))] + seq_state
                states[idx] = seq_state
        actions = [self.actions[batch_idx][seq_last_idx] for batch_idx, seq_last_idx in zip(batch_idxs, seq_last_idxs)]
        
        # TODO: 문제가 있는데 원인을 모르겠음.
        for idx, action in enumerate(actions):
            if action is None:
                states.remove(states[idx])
                actions.remove(action)
        
        # TODO: 문제있으면 해결해야할 듯? -> 각 step마다 구하는 방식으로?
        values = self.values[batch_idxs]
        returns = self.returns[batch_idxs]
        gaes = self.gaes[batch_idxs]
        gaes = (gaes - gaes.mean()) / gaes.std() + 1e-8
        vnf_s_logpas = self.vnf_s_logpas[batch_idxs]
        vnf_p_logpas = self.vnf_p_logpas[batch_idxs]

        return states, actions, returns, gaes, vnf_s_logpas, vnf_p_logpas, values

    def __len__(self):
        return len(self.actions)