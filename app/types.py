from typing import List, Literal
from dataclasses import dataclass

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


@dataclass
class SFC:
    id: int
    length: int


@dataclass
class SRV:
    id: int
    totVcpuNum: int
    totVmemMb: int
    sleepable: bool


@dataclass
class Rack:
    id: int
    srvList: List[SRV]


@dataclass
class State:
    rackList: List[Rack]
    sfcList: List[SFC]
    vnfList: List[VNF]


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
    sleepNum: int


@dataclass
class Step:
    state: State
    action: Action
    info: Info


@dataclass
class Episode:
    steps: List[Step]
