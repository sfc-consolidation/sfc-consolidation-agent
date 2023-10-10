from typing import List, Tuple, Union
from functools import reduce
from dataclasses import dataclass, fields as datafields

import torch

from app.types import VNF, SRV, Rack, State, Info


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

def calc_reward(info: Info, next_info: Info):
    curr_avg_power = sum(info.powerList) / len(info.powerList)
    next_avg_power = sum(next_info.powerList) / len(next_info.powerList)
    curr_avg_latency = sum(info.latencyList) / len(info.latencyList)
    next_avg_latency = sum(next_info.latencyList) / len(next_info.latencyList)

    reward = next_avg_power - curr_avg_power
    reward += curr_avg_latency - next_avg_latency #TODO: latency를 줄이는 것이 목표 + latency값의 범위를 0~1로 바꿔야 함
    return reward

def get_possible_action_mask(batch: Union[List[List[State]], List[State], State]):
    if isinstance(batch, State):
        batch = [[batch]]
    elif isinstance(batch, list):
        if isinstance(batch[0], State):
            batch = [[state] for state in batch]
    
    batch_size = len(batch)
    seq_len = len(batch[0])
    vnf_size = len(batch[0][0].vnfList)
    srv_size = len(batch[0][0].srvList)

    # Creating mask with ones
    mask = torch.ones((batch_size, seq_len, vnf_size, srv_size))

    # TODO: change to vector operation
    for b_idx, seq in enumerate(batch):
        for seq_idx, state in enumerate(seq):
            state = injectSrvUsage(state)
            vnfList = state.vnfList
            srvList = state.srvList
            for i, vnf in enumerate(vnfList):
                if not vnf.movable:
                    mask[b_idx, seq_idx, i, :] = 0
                for j, srv in enumerate(srvList):
                    if vnf.srvId == srv.id:
                        mask[b_idx, seq_idx, i, j] = 0
                        continue
                    if srv.totVcpuNum - srv.useVcpuNum < vnf.reqVcpuNum or srv.totVmemMb - srv.useVmemMb < vnf.reqVmemMb:
                        mask[b_idx, seq_idx, i, j] = 0
                        continue
    return mask