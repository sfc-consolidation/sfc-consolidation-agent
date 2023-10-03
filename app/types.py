from typing import List, Literal
from dataclasses import dataclass
from functools import reduce

import torch


Algorithm = Literal["dqn", "ppo", "ff", "eehvmc", "random"]


@dataclass
class VNF:
    id: int
    srvId: int
    sfcId: int
    orderInSfc: int
    reqVcpuNum: int
    reqVmemMb: int
    movable: bool

    def to_tensor(self):
        return torch.tensor([self.id, self.srvId, self.sfcId, self.orderInSfc, self.reqVcpuNum, self.reqVmemMb, self.movable])


@dataclass
class SFC:
    id: int
    length: int

    def to_tensor(self):
        return torch.tensor([self.id, self.length])


@dataclass
class SRV:
    id: int
    totVcpuNum: int
    totVmemMb: int
    sleepable: bool

    def to_tensor(self):
        return torch.tensor([self.id, self.totVcpuNum, self.totVmemMb, self.sleepable])


@dataclass
class Rack:
    id: int
    srvList: List[SRV]

    def to_tensor(self):
        return torch.tensor([self.id])


@dataclass
class State:
    rackList: List[Rack]
    sfcList: List[SFC]
    vnfList: List[VNF]

    def to_tensor(self):
        srvList: List[SRV] = list(reduce(lambda acc, x: acc +
                                         x.srvList, self.rackList, []))
        return (
            torch.stack([rack.to_tensor() for rack in self.rackList]),
            torch.stack([srv.to_tensor() for srv in srvList]),
            torch.stack([sfc.to_tensor() for sfc in self.sfcList]),
            torch.stack([vnf.to_tensor() for vnf in self.vnfList])
        )


@dataclass
class Action:
    vnfId: int
    srvId: int


@dataclass
class Info:
    power: List[float]  # Watt
    bandwidth: List[float]  # Mbps
    cpuUtil: List[float]
    memUtil: List[float]
    sleep: List[bool]
    sleepNum: int


@dataclass
class Step:
    state: State
    action: Action
    info: Info


@dataclass
class Episode:
    steps: List[Step]
