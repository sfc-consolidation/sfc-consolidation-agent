import os
import json
import requests
from dataclasses import dataclass
from typing import Optional, Tuple
import multiprocessing as mp

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

    def step(self, action: Action) -> Tuple[State, Info, bool]:
        target_address = self.get_target_address_fn()
        request_data = json.dumps({"action": action}, default=lambda o: o.__dict__)
        response = requests.post(
            url=f"http://{target_address}/step",
            headers=self.headers,
            data=request_data,
        )
        response_data = response.json()
        response_data = response.json() 
        state = utils.dataclass_from_dict(State, response_data["state"]) # TODO: -> error: when server return status 500 response
        info = utils.dataclass_from_dict(Info, response_data["info"])
        done = response_data["done"]
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
        return state, info, done

class MultiprocessEnvironment:
    def __init__(self, n_workers, make_env_fn):
        # for multiprocessing
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['OMP_NUM_THREADS'] = '1'

        self.n_workers = n_workers
        self.make_env_fn = make_env_fn

        self.pipes = [mp.Pipe() for _ in range(n_workers)]
        self.workers = [mp.Process(target=self._work, args=(rank, self.pipes[rank][1])) for rank in range(n_workers)]
        [w.start() for w in self.workers]

    def close(self, **kwargs):
        self._broadcast_msg(("close", kwargs))
        [w.join() for w in self.workers]
    
    def _send_msg(self, msg, rank):
        parent_end = self.pipes[rank][0]
        parent_end.send(msg)

    def _broadcast_msg(self, msg):
        [self._send_msg(msg, rank) for rank in range(self.n_workers)]
    
    def _work(self, rank, child_end):
        env = self.make_env_fn(rank)
        while True:
            cmd, kwargs = child_end.recv()
            if cmd == "step":
                state, info, done = env.step(**kwargs)
                child_end.send((state, info, done))
            elif cmd == "reset":
                state, info, done = env.reset(**kwargs)
                child_end.send((state, info, done))
            elif cmd == "close":
                del env
                child_end.close()
                break
            else:
                del env
                child_end.close()
                break
    
    def reset(self, ranks=None, **kwargs):
        if ranks is not None:
            [self._send_msg(("reset", kwargs), rank) for rank in ranks]
            return [self.pipes[rank][0].recv() for rank in ranks]
        else:
            self._broadcast_msg(("reset", kwargs))
            return [self.pipes[rank][0].recv() for rank in range(self.n_workers)]
        
    def step(self, actions):
        assert len(actions) == self.n_workers

        [self._send_msg(("step", {"action": action}), rank) for rank, action in enumerate(actions)]
        
        states, infos, dones = [], [], []
        for rank in range(self.n_workers):
            state, info, done = self.pipes[rank][0].recv()
            states.append(state)
            infos.append(info)
            dones.append(done)
        return states, infos, dones