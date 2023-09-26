import os
import json
import random
from datetime import datetime
from functools import reduce

from fastapi import FastAPI
from typing import get_args

from type import Algorithm, State, Action, Episode


DIR_PATH_DICT = {
    algorithm: f"data/{algorithm}" for algorithm in get_args(Algorithm)
}

for dir_path in DIR_PATH_DICT.values():
    os.makedirs(dir_path, exist_ok=True)


app = FastAPI()


@app.post("/inference")
def inference(state: State, algorithm: Algorithm) -> Action:
    action = Action(-1, -1)
    if (algorithm == "dqn"):
        return action  # TODO
    elif (algorithm == "ppo"):
        return action  # TODO
    elif (algorithm == "bf"):
        return action  # TODO
    elif (algorithm == "ff"):
        return action  # TODO
    else:
        # Randomly select vnf and srv id
        vnfNum = len(state.vnfList)
        srvNum = len(
            list(reduce(lambda acc, x: acc + x.srvList, state.rackList, [])))

        action.vnfId = random.randint(0, vnfNum - 1)
        action.srvId = random.randint(0, srvNum - 1)

    return action


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


@app.post("/simulate")
def simulate(state: State, algorithm: Algorithm) -> Episode:
    raise NotImplementedError  # TODO
