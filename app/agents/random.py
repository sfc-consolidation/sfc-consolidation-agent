import random

from app.utils import utils
from app.types import State, Action
from app.agents.agent import Agent


class RandomAgent(Agent):
    """
    RandomAgent generate randomly selected action.
    1. Stop or Do
        - Random agent always try to do.
        - But, If there is no VNF to allocate, random agent stop.
        - Also, If there is no server that can allocate a VNF, random agent stop.
    2. Select VNF
        - Select the VNF randomly.
    3. Select Server
        - Select the server randomly, but not choose the server that the VNF is already allocated.
        - If there is no server that can allocate the VNF, random agent stop.
    """

    name = "Random"

    def inference(self, state: State) -> Action:

        # Randomly select vnf and srv id
        vnfNum = len(state.vnfList)
        srvNum = len(utils.getSrvList(state.rackList))
        vnfIdx = random.randint(0, vnfNum - 1)
        srvIdx = random.randint(0, srvNum - 1)
        vnfId = state.vnfList[vnfIdx].id
        srvId = utils.getSrvList(state.rackList)[srvIdx].id
        return Action(vnfId, srvId)
