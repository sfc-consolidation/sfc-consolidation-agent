from app.types import State, Action
from app.agents.agent import Agent


class DQNAgent(Agent):
    name = "DRL (DQN)"

    @classmethod
    def inference(state: State) -> Action:
        # TODO: Implement DQN inference
        return None
