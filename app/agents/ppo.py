from app.types import State, Action
from app.agents.agent import Agent


class PPOAgent(Agent):
    name = "DRL(PPO)"

    @classmethod
    def inference(state: State) -> Action:
        # TODO: Implement PPO inference
        return None
