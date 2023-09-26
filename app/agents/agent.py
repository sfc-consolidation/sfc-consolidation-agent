from app.types import State, Action


class Agent:
    """
    Agent is an abstract class that defines the interface for all agents.
    All agents must implement the inference method.
    Every agent must have three steps:
    1. Stop or Do
        - Each agent must decide whether to stop or do.
        - In this system, if agent decide to stop, agent return None.
    2. Select VNF
        - If agent decide to do, agent must select the VNF to allocate.
    3. Select Server
        - If agent decide to do, agent must select the server to allocate.
    """

    name: str

    @classmethod
    def inference(cls: 'Agent', state: State) -> Action:
        raise NotImplementedError
