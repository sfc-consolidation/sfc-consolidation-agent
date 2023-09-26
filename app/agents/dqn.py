from app.types import State, Action
from app.agents.agent import Agent
from app.utils import utils


class DQNAgent(Agent):
    name = "DRL (DQN)"

    @classmethod
    def inference(cls: 'DQNAgent', state: State) -> Action:
        # TODO: Implement DQN inference
        return None
