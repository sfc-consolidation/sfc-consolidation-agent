from typing import List, Tuple
from functools import reduce
from dataclasses import dataclass, fields as datafields

from app.types import VNF, SRV, Rack, State


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
        cpuUsage[vnf.srvId] += vnf.reqVcpuNum
        memUsage[vnf.srvId] += vnf.reqVmemMb

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
            srv.useVcpuNum = cpuUsage[srv.id]
            srv.useVmemMb = memUsage[srv.id]
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
