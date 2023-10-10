import os
import json
from datetime import datetime

from fastapi import FastAPI
from typing import get_args, Union

from app.types import Algorithm, State, Action, Episode, Step
from app.agents.agent import Agent
from app.agents.ff import FFAgent
from app.agents.eehvmc import EEHVMCAgent
from app.agents.dqn import DQNAgent, DQNAgentInfo
from app.agents.ppo import PPOAgent, PPOAgentInfo
from app.dl.models.encoder import EncoderInfo
from app.dl.models.core import StateEncoderInfo
from app.dl.models.dqn import DQNValueInfo
from app.dl.models.ppo import PPOValueInfo, PPOPolicyInfo
from app.agents.random import RandomAgent
from app.constants import *

stateEncoderInfo = StateEncoderInfo(
    max_rack_num=MAX_RACK_NUM,
    rack_id_dim=2,
    max_srv_num=MAX_SRV_NUM,
    srv_id_dim=2,
    srv_encoder_info=EncoderInfo(
        input_size=2 + 3,
        output_size=4,
        hidden_sizes=[8],
        batch_norm=True,
        method="SA",
        dropout=0.3,
        num_head=2,
        device=TORCH_DEVICE,
    ),
    max_sfc_num=MAX_SFC_NUM,
    sfc_id_dim=4,
    sfc_encoder_info=EncoderInfo(
        input_size=4 + 1,
        output_size=4,
        hidden_sizes=[8],
        batch_norm=True,
        method="SA",
        dropout=0.3,
        num_head=2,
        device=TORCH_DEVICE,
    ),
    max_vnf_num=MAX_VNF_NUM,
    vnf_id_dim=4,
    vnf_encoder_info=EncoderInfo(
        input_size=4 + 2 + 4 + 4 + 3,
        output_size=8,
        hidden_sizes=[16],
        batch_norm=True,
        method="SA",
        dropout=0.3,
        num_head=4,
        device=TORCH_DEVICE,
    ),
    core_encoder_info=EncoderInfo(
        input_size=2 * MAX_RACK_NUM + 4 * MAX_SRV_NUM + 4 * MAX_SFC_NUM + 8 * MAX_VNF_NUM,
        output_size=8,
        hidden_sizes=[32, 16],
        batch_norm=True,
        method="LSTM",
        dropout=0.3,
        device=TORCH_DEVICE,
    ),
    device=TORCH_DEVICE,
)



agents = {
    'ff': FFAgent(),
    'eehvmc': EEHVMCAgent(),
    'random': RandomAgent(),
    'dqn': DQNAgent(DQNAgentInfo(
        encoder_info=stateEncoderInfo,
        vnf_s_value_info=DQNValueInfo(
            query_size=8,
            key_size=8,
            value_size=8,
            hidden_sizes=[8, 8],
            num_heads=[4, 4],
            dropout=0.3,
            device=TORCH_DEVICE,
        ),
        vnf_p_value_info=DQNValueInfo(
            query_size=MAX_VNF_NUM,
            key_size=4,
            value_size=4,
            hidden_sizes=[8, 8],
            num_heads=[4, 4],
            dropout=0.3,            
            device=TORCH_DEVICE,
        ),
    )),
    'ppo': PPOAgent(PPOAgentInfo(
        encoder_info=stateEncoderInfo,
        vnf_value_info=PPOValueInfo(
            query_size=8,
            key_size=8,
            value_size=8,
            hidden_sizes=[8, 8],
            num_heads=[4, 4],
            dropout=0.3,
            seq_len=MAX_VNF_NUM,
            device=TORCH_DEVICE,
        ),
        vnf_s_policy_info=PPOPolicyInfo(
            query_size=8,
            key_size=8,
            value_size=8,
            hidden_sizes=[8, 8],
            num_heads=[4, 4],
            dropout=0.3,
            device=TORCH_DEVICE,
        ),
        vnf_p_policy_info=PPOPolicyInfo(
            query_size=MAX_VNF_NUM,
            key_size=4,
            value_size=4,
            hidden_sizes=[8, 8],
            num_heads=[4, 4],
            dropout=0.3,
            device=TORCH_DEVICE,
        ),
    )),
}


EPISODE_DIR_PATH_DICT = {
    algorithm: f"data/episode/{algorithm}" for algorithm in get_args(Algorithm)
}

for dir_path in EPISODE_DIR_PATH_DICT.values():
    os.makedirs(dir_path, exist_ok=True)

STEP_DIR_PATH_DICT = {
    algorithm: f"data/step/{algorithm}" for algorithm in get_args(Algorithm)
}

for dir_path in STEP_DIR_PATH_DICT.values():
    os.makedirs(dir_path, exist_ok=True)

app = FastAPI()


@app.post("/inference")
def inference(state: State, algorithm: Algorithm) -> Union[Action, None]:
    agent: Agent = None
    if (algorithm == "ff"):
        agent = agents["ff"]
    elif (algorithm == "eehvmc"):
        agent = agents["eehvmc"]
    elif (algorithm == "dqn"):
        agent = agents["dqn"]
    elif (algorithm == "ppo"):
        agent = agents["ppo"]
    else:
        agent = agents["random"]
    return agent.inference(state)


@app.post("/save-episode")
def save_episode(episode: Episode, algorithm: Algorithm):
    # Save Episode as a file
    dir_path = EPISODE_DIR_PATH_DICT[algorithm]
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

@app.post("/save-step")
def save_step(step: Step, algorithm: Algorithm, id: str):
    dir_path = STEP_DIR_PATH_DICT[algorithm]
    file_name = f"{id}.json"
    path = os.path.join(dir_path, file_name)

    with open(path, "w") as f:
        json.dump(step, f, default=lambda o: o.__dict__)
