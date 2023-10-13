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



class RewardStandardization:
    mean = 0
    M2 = 1
    n = 0
    
    @classmethod
    def update(cls, data):
        cls.n += 1
        delta = data - cls.mean
        cls.mean += delta / cls.n
        delta2 = data - cls.mean
        cls.M2 += delta * delta2
    
    @classmethod
    def scale(cls, data):
        cls.update(data)
        mean = cls.mean
        var = cls.M2 / (cls.n - 1) if cls.n > 1 else 1
        std = var ** 0.5
        return (data - mean) / std

def calc_reward(ini_info: Info, fin_info: Info):
    ini_avg_power = sum(ini_info.powerList) / len(ini_info.powerList)
    fin_avg_power = sum(fin_info.powerList) / len(fin_info.powerList)
    ini_avg_latency = sum(ini_info.latencyList) / len(ini_info.latencyList)
    fin_avg_latency = sum(fin_info.latencyList) / len(fin_info.latencyList)

    reward = ini_avg_power - fin_avg_power
    reward += ini_avg_latency - fin_avg_latency
    reward = RewardStandardization.scale(reward)
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