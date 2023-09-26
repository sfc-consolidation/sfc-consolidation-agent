from typing import List

from app.agents.agent import Agent
from app.types import State, Action
from app.utils import utils


class FFAgent(Agent):
    """
    FFAgent is First Fit algorithm based agent.
    (Please check https://github.com/sfc-consolidation/sfc-consolidation-agent/issues/3)
    1. Stop or Do
        - FF agent always try to do.
        - But, if there is no VNF to allocate, FF agent stop.
        - Also, if there is no server that can allocate a VNF, FF agent stop.
    2. Select VNF
        - First, FF agent selects the Server that lowest resource usage.
        - After that, FF agent selects the VNF that highest resource usage. (before that, FF agent checks the VNF is movable)
    3. Select Server
        - FF agent selects the Server that highest resource usage, that also have capacity to allocate the VNF.
    """
    name = "First Fit"

    @classmethod
    def inference(cls: 'FFAgent', state: State) -> Action:
        utils.injectSrvUsage(state)
        sorted_srv_idxs = cls._get_sorted_srv_idxs_with_srv_load(state)
        while len(sorted_srv_idxs) > 0:
            src_srv_id = cls._select_and_pop_srv(sorted_srv_idxs)
            sorted_vnf_idxs = cls._get_sorted_vnf_idxs_with_vnf_req(
                state, src_srv_id)
            while len(sorted_vnf_idxs) > 0:
                vnf_id = cls._select_and_pop_vnf(sorted_vnf_idxs)
                possible_tgt_srv_idxs = cls._get_possible_tgt_srv_idxs_with_srv_load(
                    state, src_srv_id, vnf_id)
                tgt_srv_id = cls._place_vnf(possible_tgt_srv_idxs)
                if tgt_srv_id is not None:
                    return Action(vnf_id, tgt_srv_id)
        return None

    @classmethod
    def _get_sorted_srv_idxs_with_srv_load(cls: 'FFAgent', state: State) -> List[int]:
        srv_loads = []
        for srv in state.srvList:
            srv_load = srv.useVcpuNum
            srv_loads.append((srv.id, srv_load))
        sorted_srv_loads = sorted(srv_loads, key=lambda x: x[1])
        sorted_srv_idxs = [x[0] for x in sorted_srv_loads]
        return sorted_srv_idxs

    @classmethod
    def _get_sorted_vnf_idxs_with_vnf_req(cls: 'FFAgent', state: State, src_srv_id: int) -> List[int]:
        vnf_reqs = []
        for vnf in state.vnfList:
            if vnf.srvId == src_srv_id and vnf.movable:
                vnf_req = vnf.reqVcpuNum
                vnf_reqs.append((vnf.id, vnf_req))
        sorted_vnf_reqs = sorted(vnf_reqs, key=lambda x: x[1])
        sorted_vnf_idxs = [x[0] for x in sorted_vnf_reqs]
        return sorted_vnf_idxs

    @classmethod
    def _get_possible_tgt_srv_idxs_with_srv_load(cls: 'FFAgent', state: State, src_srv_id: int, vnf_id: int) -> List[int]:
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
        possible_tgt_srv_idxs = sorted(possible_tgt_srv_idxs, key=lambda x: state.srvList[x].useVcpuNum /
                                       state.srvList[x].totVcpuNum + state.srvList[x].useVmemMb / state.srvList[x].totVmemMb)
        return possible_tgt_srv_idxs

    @classmethod
    def _select_and_pop_srv(cls: 'FFAgent', sorted_srv_idxs: List[int]) -> int:
        min_load_srv_idx = sorted_srv_idxs.pop(0)
        return min_load_srv_idx

    @classmethod
    def _select_and_pop_vnf(cls: 'FFAgent', sorted_vnf_idxs: List[int]) -> int:
        min_req_vnf_idx = sorted_vnf_idxs.pop(0)
        return min_req_vnf_idx

    @classmethod
    def _place_vnf(cls: 'FFAgent', possible_tgt_srv_idxs: List[int]) -> int:
        if len(possible_tgt_srv_idxs) == 0:
            return None
        tgt_srv_id = possible_tgt_srv_idxs.pop(-1)
        return tgt_srv_id
