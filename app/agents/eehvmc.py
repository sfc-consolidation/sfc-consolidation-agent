from app.types import State, Action
from app.agents.agent import Agent


class EEHVMCAgent(Agent):
    name = "Energy Efficient Heuristic VM Consolidation"

    @classmethod
    def inference(state: State) -> Action:
        # TODO: Implement EEHVMC inference
        return None
