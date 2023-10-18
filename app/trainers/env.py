import os
import json
import asyncio
import aiohttp
import requests
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch.multiprocessing as mp

from app.utils import utils
from app.types import Action, State, Info


@dataclass
class ResetArg:
    maxRackNum: int
    minRackNum: int
    maxSrvNumInSingleRack: int
    minSrvNumInSingleRack: int
    maxVnfNum: int
    minVnfNum: int
    maxSfcNum: int
    minSfcNum: int
    maxSrvVcpuNum: int
    minSrvVcpuNum: int
    maxSrvVmemMb: int
    minSrvVmemMb: int
    maxVnfVcpuNum: int
    minVnfVcpuNum: int
    maxVnfVmemMb: int
    minVnfVmemMb: int

class Environment:
    def __init__(self, get_target_address_fn):
        self.get_target_address_fn = get_target_address_fn
        self.headers = {"Content-Type": "application/json"}
        self.cur_step = 0

    def step(self, action: Action) -> Tuple[State, Info, bool]:
        self.cur_step += 1
        target_address = self.get_target_address_fn()
        request_data = json.dumps({"action": action}, default=lambda o: o.__dict__)
        response = requests.post(
            url=f"http://{target_address}/step",
            headers=self.headers,
            data=request_data,
        )
        response_data = response.json()
        state = utils.dataclass_from_dict(State, response_data["state"]) # TODO: -> error: when server return status 500 response
        info = utils.dataclass_from_dict(Info, response_data["info"])
        done = response_data["done"] or self.cur_step >= self.max_step
        return state, info, done
    
    def reset(self, resetArg: Optional[ResetArg] = None) -> Tuple[State, Info, bool]:
        target_address = self.get_target_address_fn()
        request_data = json.dumps({"resetArg": resetArg}, default=lambda o: o.__dict__)
        response = requests.post(
            url=f"http://{target_address}/reset",
            headers=self.headers,
            data=request_data,
        )
        response_data = response.json() 
        state = utils.dataclass_from_dict(State, response_data["state"])
        info = utils.dataclass_from_dict(Info, response_data["info"])
        done = response_data["done"]

        self.cur_step = 0
        self.max_step = len(state.vnfList)

        return state, info, done


class AsyncEnvironment:
    def __init__(self, get_target_address_fn):
        self.get_target_address_fn = get_target_address_fn
        self.headers = {"Content-Type": "application/json"}
        self.cur_step = 0
    
    async def step(self, session: aiohttp.ClientSession, action: Action) -> Tuple[State, Info, bool]:
        self.cur_step += 1
        target_address = self.get_target_address_fn()
        request_data = json.dumps({"action": action}, default=lambda o: o.__dict__)
        async with session.post(
            url=f"http://{target_address}/step",
            headers=self.headers,
            data=request_data,
        ) as response:
            response_data = await response.json()
            state = utils.dataclass_from_dict(State, response_data["state"])
            info = utils.dataclass_from_dict(Info, response_data["info"])
            done = response_data["done"] or self.cur_step >= self.max_step
            return state, info, done
    
    async def reset(self, session: aiohttp.ClientSession, resetArg: Optional[ResetArg] = None) -> Tuple[State, Info, bool]:
        target_address = self.get_target_address_fn()
        request_data = json.dumps({"resetArg": resetArg}, default=lambda o: o.__dict__)
        async with session.post(
            url=f"http://{target_address}/reset",
            headers=self.headers,
            data=request_data,
        ) as response:
            response_data = await response.json()
            state = utils.dataclass_from_dict(State, response_data["state"])
            info = utils.dataclass_from_dict(Info, response_data["info"])
            done = response_data["done"]

            self.cur_step = 0
            self.max_step = len(state.vnfList)

            return state, info, done

# reset: 특정 대상만 지정해서 reset이 가능해야 함.
# step: 모든 대상을 한 번에 step할 것임.
# asyncio 이용해서 적용하는 것 수행할 것.
class MultiprocessEnvironment:
    def __init__(self, n_workers, make_env_fn):
        self.n_workers = n_workers
        self.make_env_fn = make_env_fn
        self.envs: List[AsyncEnvironment] = [make_env_fn(rank, is_async=True) for rank in range(n_workers)]
    
    async def reset(self, ranks=None, resetArg: Optional[ResetArg] = None):
        if ranks is None:
            ranks = [rank for rank in range(self.n_workers)]
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(*[self.envs[rank].reset(session, resetArg) for rank in ranks])
            states, infos, dones = [], [], []
            for i in range(len(ranks)):
                state, info, done = results[i]
                states.append(state)
                infos.append(info)
                dones.append(done)
            return states, infos, dones
        
    async def step(self, actions):
        assert len(actions) == self.n_workers

        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(*[self.envs[rank].step(session, action) for rank, action in enumerate(actions)])
            states, infos, dones = [], [], []
            for rank in range(self.n_workers):
                state, info, done = results[rank]
                states.append(state)
                infos.append(info)
                dones.append(done)
            return states, infos, dones