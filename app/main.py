import os
import json
from datetime import datetime

from fastapi import FastAPI
from typing import get_args

from app.types import Algorithm, State, Action, Episode
from app.agents.agent import Agent
from app.agents.ff import FFAgent
from app.agents.eehvmc import EEHVMCAgent
from app.agents.dqn import DQNAgent
from app.agents.ppo import PPOAgent
from app.agents.random import RandomAgent

DIR_PATH_DICT = {
    algorithm: f"data/{algorithm}" for algorithm in get_args(Algorithm)
}

for dir_path in DIR_PATH_DICT.values():
    os.makedirs(dir_path, exist_ok=True)


app = FastAPI()


@app.post("/inference")
def inference(state: State, algorithm: Algorithm) -> Action:
    agent = Agent
    if (algorithm == "ff"):
        agent = FFAgent()
    elif (algorithm == "eehvmc"):
        agent = EEHVMCAgent()
    elif (algorithm == "dqn"):
        agent = DQNAgent()
    elif (algorithm == "ppo"):
        agent = PPOAgent()
    else:
        agent = RandomAgent()
    return agent.inference(state)


@app.post("/save-episode")
def save(episode: Episode, algorithm: Algorithm):
    # Save Episode as a file
    dir_path = DIR_PATH_DICT[algorithm]
    # get current time
    cur_time = datetime.now().strftime("%Y%m%d%H%M%S")
    rack_num = len(episode.steps[0].state.rackList)
    srv_num = len(episode.steps[0].state.rackList[0].srvList)
    vnf_num = len(episode.steps[0].state.vnfList)
    sfc_num = len(episode.steps[0].state.sfcList)

    file_name = f"{rack_num}_{srv_num}_{vnf_num}_{sfc_num}_{cur_time}.json"
    path = os.path.join(dir_path, file_name)

    # write episode class to json file
    # that dosen't have __dict__ attribute
    # so, can't use dict() function
    with open(path, "w") as f:
        json.dump(episode, f, default=lambda o: o.__dict__)
