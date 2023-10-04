import os
import json
from typing import List, Tuple
from functools import reduce

from app.types import State, Action, Episode
from app.agents.agent import Agent
from app.utils.utils import injectSrvUsage, dataclass_from_dict


class EEHVMCAgent(Agent):
    """
    EEHVMCAgent is Energy Efficient Heuristic VM Consolidation algorithm based agent.
    (Please check https://github.com/sfc-consolidation/sfc-consolidation-agent/issues/1)
    """
    name = "Energy Efficient Heuristic VM Consolidation"

    def inference(self, state: State) -> Action:
        T_cpu_low, T_cpu_high, T_mem_low, T_mem_high = self._getThreshold()
        injectSrvUsage(state)

        m = len(state.srvList)

        holList = []
        hulList = []
        hmlList = []

        for i in range(m):
            srv = state.srvList[i]
            srv.cU_H = srv.useVcpuNum / srv.totVcpuNum
            srv.mU_H = srv.useVmemMb / srv.totVmemMb

            # Host Over Loaded(HOL)
            if srv.cU_H >= T_cpu_high and srv.mU_H >= T_mem_high:
                holList.append(srv)
            # Host Under Loaded(HUL)
            elif srv.cU_H <= T_cpu_low and srv.mU_H <= T_mem_low:
                hulList.append(srv)
            # Host Medium Loaded(HML)
            else:
                hmlList.append(srv)

        # This is the one of difference with FF.
        # probably HUL to HML is more important than HOL to HML.
        # so, in this system, sort each HUL and HOL, then concat them.
        hulList = sorted(hulList, key=lambda x: x.cU_H)
        holList = sorted(holList, key=lambda x: x.cU_H)
        hmlList = sorted(hmlList, key=lambda x: x.cU_H)

        candidates = hulList + holList

        # candidates Host -> HML
        #   - select and pop HUL
        #   - select vnf
        #   - select and pop from possible HML
        #   - select server
        #   - return Action
        while len(candidates) > 0:
            srv = candidates.pop(0)
            sorted_vnf_idxs = self._get_sorted_vnf_idxs_with_vnf_req(
                state, srv.id)
            while len(sorted_vnf_idxs) > 0:
                vnf_id = sorted_vnf_idxs.pop(0)
                possible_tgt_srv_idxs = self._get_possible_tgt_hml_idxs_with_srv_load(
                    state, srv.id, vnf_id)
                tgt_srv_id = self._place_vnf(possible_tgt_srv_idxs)
                if tgt_srv_id is not None:
                    return Action(vnf_id, tgt_srv_id)
        return None

    def _getThreshold(self) -> Tuple[float, float]:
        utils = []
        # 1. read files from data/random/**.json
        # get all .json file names
        pathes = os.listdir("data/random")
        # read all .json files
        for path in pathes:
            path = "data/random/" + path
            with open(path, "r") as f:
                episode = json.load(f)
                episode = dataclass_from_dict(Episode, episode)
                stateList = list(map(lambda step: step.state, episode.steps))
                stateList = list(
                    map(lambda state: injectSrvUsage(state), stateList))
                utils += reduce(self._reduceUtils, stateList, [])
        cpuUtils = list(map(lambda util: util[0], utils))
        memUtils = list(map(lambda util: util[1], utils))
        if len(cpuUtils) == 0 or len(memUtils) == 0:
            return 0.25, 0.75, 0.25, 0.75
        # 2. calculate T_high, T_low (IQR)
        # 2-1. sort utils
        cpuUtils = sorted(cpuUtils)
        memUtils = sorted(memUtils)
        # 2-2. calculate Q1, Q3
        cpuQ1 = cpuUtils[int(len(cpuUtils) * 0.25)]
        cpuQ3 = cpuUtils[int(len(cpuUtils) * 0.75)]
        memQ1 = memUtils[int(len(memUtils) * 0.25)]
        memQ3 = memUtils[int(len(memUtils) * 0.75)]

        return cpuQ1, cpuQ3, memQ1, memQ3

    @staticmethod
    def _reduceUtils(acc: List[Tuple[float, float]], state: State) -> List[Tuple[float, float]]:
        acc += [(srv.useVcpuNum / srv.totVcpuNum,
                srv.useVmemMb / srv.totVmemMb) for srv in state.srvList]
        return acc

    def _get_sorted_vnf_idxs_with_vnf_req(self, state: State, src_srv_id: int) -> List[int]:
        vnf_reqs = []
        for vnf in state.vnfList:
            if vnf.srvId == src_srv_id and vnf.movable:
                # * This is the one of difference with FF. (MRCU) *
                vnf_req = vnf.reqVcpuNum / vnf.reqVmemMb
                vnf_reqs.append((vnf.id, vnf_req))
        sorted_vnf_reqs = sorted(vnf_reqs, key=lambda x: x[1])
        sorted_vnf_idxs = [x[0] for x in sorted_vnf_reqs]
        return sorted_vnf_idxs

    def _get_possible_tgt_hml_idxs_with_srv_load(self, state: State, src_srv_id: int, vnf_id: int) -> List[int]:
        # find vnf
        for v in state.vnfList:
            if v.id == vnf_id:
                vnf = v
                break
        possible_tgt_srv_idxs = []
        for tgt_srv in state.srvList:
            if tgt_srv.id == src_srv_id:
                continue
            if tgt_srv.useVcpuNum + vnf.reqVcpuNum > tgt_srv.totVcpuNum:
                continue
            if tgt_srv.useVmemMb + vnf.reqVmemMb > tgt_srv.totVmemMb:
                continue
            possible_tgt_srv_idxs.append(tgt_srv.id)
        possible_tgt_srv_idxs = sorted(possible_tgt_srv_idxs, key=lambda x: state.srvList[x - 1].useVcpuNum /
                                       state.srvList[x - 1].totVcpuNum + state.srvList[x - 1].useVmemMb / state.srvList[x - 1].totVmemMb)
        return possible_tgt_srv_idxs

    def _place_vnf(self, possible_tgt_srv_idxs: List[int]) -> int:
        if len(possible_tgt_srv_idxs) == 0:
            return None
        tgt_srv_id = possible_tgt_srv_idxs.pop(-1)
        return tgt_srv_id
