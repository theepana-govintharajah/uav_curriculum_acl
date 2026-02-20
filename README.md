\
# UAV Curriculum ACL (Energy + Comms + Dynamic Tasks) — Ready-to-run Python Code

This repo implements a **curriculum learning pipeline** that progresses through your 6 steps:

1) **Single UAV navigation** (unlimited energy)  
2) **UAV swarm** navigation with **dynamic obstacles**  
3) **Cooperative task allocation** (static tasks)  
4) **Dynamic task arrival**  
5) **Energy budget tightness**  
6) **Communication sparsity**

It includes baselines to answer your key question:

> Can resource-aware curriculum improve final swarm performance under strict energy and communication limits?

## What you get
- Custom **Gymnasium** single-agent env: `SingleUAVNavEnv`
- Custom **PettingZoo ParallelEnv** multi-agent env: `UAVSwarmTaskEnv`
- A simple **parameter-sharing PPO** trainer (PyTorch) that can train:
  - single-agent env
  - multi-agent env (treats each agent as a sample, shared policy across agents)
- A curriculum manager:
  - **resource-aware adaptive schedule** (keeps success-rate in a target band)
  - baseline schedules: **no curriculum (hard from start)**, **linear**, **random**
- Evaluation + plotting scripts

## Install (Windows)
From this folder:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart
Train the curriculum agent (resource-aware):
```bash
python scripts/train.py --schedule resource_aware --seed 0
```

Train baselines:
```bash
python scripts/train.py --schedule none --seed 0
python scripts/train.py --schedule linear --seed 0
python scripts/train.py --schedule random --seed 0
```

Evaluate and compare:
```bash
python scripts/eval.py --runs_dir runs
python scripts/plot.py --runs_dir runs
```

Optional rendering (slow):
```bash
python scripts/play.py --ckpt runs/resource_aware_seed0/checkpoints/latest.pt --mode swarm
```

## Notes (important)
- This is designed to run on a normal laptop CPU. Keep defaults first.
- The environments are **not flight-controller accurate** (by design); they incorporate:
  - continuous dynamics + drag
  - energy usage model
  - wind field
  - dynamic obstacles
  - comm-range-limited neighbor info
- You can later swap the env dynamics to MuJoCo/Flightmare/PX4 while keeping the curriculum logic.

## Folder structure
- `uav_acl/` core code
  - `envs/` environments
  - `rl/` PPO trainer + utils
  - `curriculum/` schedule logic
- `scripts/` entry points
- `runs/` outputs (created after training)

