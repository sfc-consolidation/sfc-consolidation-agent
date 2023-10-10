## Agent

This project include 2 main features.

1. Agents
  - [x] Random Agent
  - [x] Rule-based Agent (First Fit)
  - [x] Heuristic-based Agent (EEHVMC)
  - [ ] RL-based Agent (DQN)
  - [ ] RL-based Agent (PPO)
2. REST API Server
  - [x] inference     - for getting action from agent with given state
  - [x] save-episode  - for storing episode data

### Run

```bash
$ conda create -n sfc-consolidation python=3.8.16
$ conda activate sfc-consolidation
$ conda install -c conda-forge poetry
$ poetry install
$ poetry run uvicorn app.main:app
```


### Dev

```bash
$ conda create -n sfc-consolidation python=3.8.16
$ conda activate sfc-consolidation
$ poetry install
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia # install pytorch <-- change it to your version
$ poetry run uvicorn app.main:app --reload
```
