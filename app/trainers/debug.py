import numpy as np

from app.types import State, Info


class Debugger:
    def __init__(self):
        self.ini_states = []
        self.ini_infos = []
        self.fin_states = []
        self.fin_infos = []

        self.episode_lens = []
        self.explore_rates = []

        self.srv_n = [] # total server number
        self.ini_sleep_srv_n = [] # initial total sleep server number.
        self.fin_sleep_srv_n = [] # final total sleep server number.
        self.chn_sleep_srv_n = [] # change total sleep server number.
        self.ini_latencies = [] # initial average latency of SFCs in the edge.
        self.fin_latencies = [] # final average latency of SFCs in the edge.
        self.chn_latencies = [] # change average latency of SFCs in the edge.
        self.ini_powers = [] # initial average power usage of Servers in the edge.
        self.fin_powers = [] # final average power usage of Servers in the edge.
        self.chn_powers = [] # change average power usage of Servers in the edge.

    def add_episode(self, ini_state: State, ini_info: Info, fin_state: State, fin_info: Info, explore_rate: float, episode_len: int):
        self.ini_states.append(ini_state)
        self.ini_infos.append(ini_info)
        self.fin_states.append(fin_state)
        self.fin_infos.append(fin_info)
        self.explore_rates.append(explore_rate)
        self.episode_lens.append(episode_len)

        self.srv_n.append(len(ini_info.sleepList))
        self.ini_sleep_srv_n.append(ini_info.sleepList.count(True))
        self.fin_sleep_srv_n.append(fin_info.sleepList.count(True))
        self.chn_sleep_srv_n.append(self.fin_sleep_srv_n[-1] - self.ini_sleep_srv_n[-1])
        
        ini_info.powerList = [ 0 if power == 'NaN' else power for power in ini_info.powerList ]
        fin_info.powerList = [ 0 if power == 'NaN' else power for power in fin_info.powerList ]

        self.ini_latencies.append(np.mean(ini_info.latencyList))
        self.ini_powers.append(np.mean(ini_info.powerList))

        self.fin_latencies.append(np.mean(fin_info.latencyList))
        self.fin_powers.append(np.mean(fin_info.powerList))

        self.chn_latencies.append(self.fin_latencies[-1] - self.ini_latencies[-1])
        self.chn_powers.append(self.fin_powers[-1] - self.ini_powers[-1])

    # print debugging info by last N episode's average.
    # format: Table
    # | Avg Episode Len | Avg Explore Rate | Avg Edge CPU Load | Avg Latency Chnage (Initial -> Final) | Avg Power Change (Initial -> Final) |
    # if refresh is True, then clear previous data.

    def print(self, last_n: int = 100, refresh=True):
        print("Episode Info")
        print(
            "| Avg Episode Len | Avg Explore Rate | Avg Sleep Change (Initial -> Final) | Avg Latency Change (Initial -> Final) | Avg Power Change (Initial -> Final) |"
        )
        print(
            "|-----------------|------------------|-------------------------------------|---------------------------------------|-------------------------------------|"
        )
        print(
            "| {:>15.2f} | {:>16.2f} | {:>6.2f} ({:>6.2f} -> {:>6.2f} / {:>7.2f}) | {:>10.2f} ({:>10.2f} -> {:>10.2f}) | {:>8.2f} ({:>10.2f} -> {:>10.2f}) |".format(
                np.mean(self.episode_lens[-last_n:]),
                np.mean(self.explore_rates[-last_n:]),
                np.mean(self.chn_sleep_srv_n[-last_n:]),
                np.mean(self.ini_sleep_srv_n[-last_n:]),
                np.mean(self.fin_sleep_srv_n[-last_n:]),
                np.mean(self.srv_n[-last_n:]),
                np.mean(self.chn_latencies[-last_n:]),
                np.mean(self.ini_latencies[-last_n:]),
                np.mean(self.fin_latencies[-last_n:]),
                np.mean(self.chn_powers[-last_n:]),
                np.mean(self.ini_powers[-last_n:]),
                np.mean(self.fin_powers[-last_n:]),
            )
        )