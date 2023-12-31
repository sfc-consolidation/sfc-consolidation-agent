import os
from typing import List, Tuple, Union
from functools import reduce
from dataclasses import dataclass, fields as datafields

import torch

from app.types import VNF, SRV, Rack, State, Info
from app.constants import *

def getSrvUsage(vnfList: List[VNF], srvNum: int) -> Tuple[List[int], List[int]]:
    """
    Return Each Server's CPU and MEM Usage

    Args:
        vnfList (List[VNF]): VNF List
        srvNum (int): Server Number

    Returns:
        Tuple[List[int], List[int]]: CPU Usage, MEM Usage
    """
    cpuUsage = [0] * srvNum
    memUsage = [0] * srvNum
    for vnf in vnfList:
        cpuUsage[vnf.srvId - 1] += vnf.reqVcpuNum
        memUsage[vnf.srvId - 1] += vnf.reqVmemMb

    return cpuUsage, memUsage


def getSrvList(rackList: List[Rack]) -> List[SRV]:
    """
    Get Server List from Rack List

    Args:
        rackList (List[Rack]): Rack List

    Returns:
        List[SRV]: Server List
    """
    return list(reduce(lambda acc, x: acc + x.srvList, rackList, []))


def injectSrvUsage(state: State) -> State:
    """
    Inject Each Server's CPU and MEM Usage

    Args:
        state (State): State

    Returns:
        State: State
    """
    srvList = getSrvList(state.rackList)
    cpuUsage, memUsage = getSrvUsage(state.vnfList, len(srvList))
    for rack in state.rackList:
        for srv in rack.srvList:
            srv.useVcpuNum = cpuUsage[srv.id - 1]
            srv.useVmemMb = memUsage[srv.id - 1]
    state.srvList = srvList

    return state


def dataclass_from_dict(klass, dikt):
    try:
        fieldtypes = klass.__annotations__
        return klass(**{f: dataclass_from_dict(fieldtypes[f], dikt[f]) for f in dikt})
    except AttributeError:
        if isinstance(dikt, (tuple, list)):
            return [dataclass_from_dict(klass.__args__[0], f) for f in dikt]

        return dikt


def logit_to_prob(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert Logit to Probability

    Args:
        logits (torch.Tensor): Logit (BATCH_LEN, FEATURE)

    Returns:
        torch.Tensor: Probability
    """
    probs = torch.zeros_like(logits)
    for i in range(logits.shape[0]):
        probs[i, logits[i] != 0] = torch.softmax(
            logits[i, logits[i] != 0], dim=0)
    return probs


def get_info_from_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get Action Info from Logits

    Args:
        logits (torch.Tensor): Logit (BATCH_LEN, FEATURE)

    Returns:
        torch.Tensor: Action
        torch.Tensor: Log Probability of Actions
        torch.Tensor: Is Exploration
    """
    probs = logit_to_prob(logits)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample().to(torch.int32)
    logpas = dist.log_prob(action)
    is_exploration = action != torch.argmax(probs, dim=1)
    return action, logpas, is_exploration

class RewardZscoreStandardization:
    power_mean = 0
    power_M2 = 1
    latency_mean = 0
    latency_M2 = 1
    n = 0
    
    @classmethod
    def update(cls, power_reduction, latency_reduction):
        cls.n += 1
        power_delta = power_reduction - cls.power_mean
        cls.power_mean += power_delta / cls.n
        power_delta2 = power_reduction - cls.power_mean
        cls.power_M2 += power_delta * power_delta2

        latency_delta = latency_reduction - cls.latency_mean
        cls.latency_mean += latency_delta / cls.n
        latency_delta2 = latency_reduction - cls.latency_mean
        cls.latency_M2 += latency_delta * latency_delta2
    
    @classmethod
    def scale(cls, power_reduction, latency_reduction):
        cls.update(power_reduction, latency_reduction)
        power_mean = cls.power_mean
        power_var = cls.power_M2 / (cls.n - 1) if cls.n > 1 else 1
        power_std = power_var ** 0.5
        power_reduction = (power_reduction - power_mean) / power_std

        latency_mean = cls.latency_mean
        latency_var = cls.latency_M2 / (cls.n - 1) if cls.n > 1 else 1
        latency_std = latency_var ** 0.5
        latency_reduction = (latency_reduction - latency_mean) / latency_std

        return power_reduction + latency_reduction

class RewardMinMaxStandardization:
    power_min = -1
    power_max = 1
    latency_min = -1
    latency_max = 1
    
    @classmethod
    def update(cls, power_reduction, latency_reduction):
        cls.power_min = min(cls.power_min, power_reduction)
        cls.power_max = max(cls.power_max, power_reduction)
        cls.latency_min = min(cls.latency_min, latency_reduction)
        cls.latency_max = max(cls.latency_max, latency_reduction)
    
    # -1 ~ 1 사이로 scaling
    @classmethod
    def scale(cls, power_reduction, latency_reduction):
        cls.update(power_reduction, latency_reduction)
        power_reduction = (power_reduction - cls.power_min) / (cls.power_max - cls.power_min) * 2 - 1
        latency_reduction = (latency_reduction - cls.latency_min) / (cls.latency_max - cls.latency_min) * 2 - 1

        return power_reduction * 0.9 + latency_reduction * 0.1

def calc_reward(info: Info, next_info: Info):
    avg_power = sum(info.powerList) / len(info.powerList)
    next_avg_power = sum(next_info.powerList) / len(next_info.powerList)
    avg_latency = sum(info.latencyList) / len(info.latencyList)
    next_avg_latency = sum(next_info.latencyList) / len(next_info.latencyList)

    power_reduction = avg_power - next_avg_power
    latency_reduction = avg_latency - next_avg_latency
    reward = RewardMinMaxStandardization.scale(power_reduction, latency_reduction)
    return reward

def get_possible_action_mask(batch: Union[List[List[State]], List[State], State]):
    if isinstance(batch, State):
        batch = [[batch]]
    elif isinstance(batch, list):
        if isinstance(batch[-1], State):
            batch = [[state] for state in batch]
    
    batch_size = len(batch)
    seq_len = len(batch[0])

    # Creating mask with zeros
    mask = torch.zeros((batch_size, MAX_VNF_NUM, MAX_SRV_NUM))

    # TODO: change to vector operation
    for b_idx, seq in enumerate(batch):
        state = seq[-1]
        if state is None:
            continue
        state = injectSrvUsage(state)
        vnfList = state.vnfList
        srvList = state.srvList
        for i, vnf in enumerate(vnfList):
            for j, srv in enumerate(srvList):
                if vnf.srvId == srv.id:
                    continue
                if vnf.movable and srv.totVcpuNum - srv.useVcpuNum >= vnf.reqVcpuNum and srv.totVmemMb - srv.useVmemMb >= vnf.reqVmemMb:
                    mask[b_idx, i, j] = 1
    return mask
