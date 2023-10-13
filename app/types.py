from typing import List, Literal, Union
from dataclasses import dataclass
from functools import reduce

import torch

from app.constants import *


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
        return torch.tensor([self.id, self.srvId, self.sfcId, self.orderInSfc, self.reqVcpuNum / MAX_VNF_VCPU_NUM, self.reqVmemMb / MAX_VNF_VMEM_MB, self.movable])


@dataclass
class SFC:
    id: int
    length: int

    def to_tensor(self):
        return torch.tensor([self.id, self.length / MAX_SFC_NUM])


@dataclass
class SRV:
    id: int
    totVcpuNum: int
    totVmemMb: int
    sleepable: bool

    def to_tensor(self, rackId: int = None):
        if not rackId == None:
            return torch.tensor([self.id, rackId, self.totVcpuNum / MAX_SRV_VCPU_NUM, self.totVmemMb / MAX_SRV_VMEM_MB, self.sleepable])
        return torch.tensor([self.id, self.totVcpuNum / MAX_SRV_VCPU_NUM, self.totVmemMb / MAX_SRV_VMEM_MB, self.sleepable])


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
        return (
            torch.stack([rack.to_tensor() for rack in self.rackList]),
            torch.concat([torch.stack([srv.to_tensor(rackId=rack.id) for srv in rack.srvList]) for rack in self.rackList]),
            torch.stack([sfc.to_tensor() for sfc in self.sfcList]),
            torch.stack([vnf.to_tensor() for vnf in self.vnfList])
        )

    def get_srvList(self):
        return list(reduce(lambda acc, x: acc +
                           x.srvList, self.rackList, []))


@dataclass
class Action:
    vnfId: int
    srvId: int


@dataclass
class Info:
    powerList: List[float]  # Watt
    cpuUtilList: List[float]
    memUtilList: List[float]
    bwUtilList: List[float]  # Mbps
    sleepList: List[bool]
    latencyList: List[float]
    success: bool
    sleepNum: int


@dataclass
class Step:
    state: State
    info: Info
    action: Union[Action, None] = None


@dataclass
class Episode:
    steps: List[Step]
