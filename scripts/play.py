import argparse, os
import numpy as np
import torch

from uav_acl.config import EnvConfig
from uav_acl.envs import SingleUAVNavEnv, UAVSwarmTaskEnv
from uav_acl.rl.ppo_shared import PPOShared

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--mode", type=str, default="swarm", choices=["single","swarm"])
    args = ap.parse_args()

    if args.mode == "single":
        cfg = EnvConfig(n_agents=1, energy_enabled=False, wind_enabled=False)
        env = SingleUAVNavEnv(cfg, render_mode="human", seed=0)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    else:
        cfg = EnvConfig(
            n_agents=4,
            n_obstacles=6,
            dynamic_obstacles=True,
            n_tasks=6,
            dynamic_task_arrival=True,
            energy_enabled=True,
            energy_budget=25.0,
            wind_enabled=True,
            wind_strength=0.6,
            comm_enabled=True,
            comm_range=3.5,
            packet_drop_prob=0.15,
            comm_delay_steps=1,
        )
        env = UAVSwarmTaskEnv(cfg, seed=0)
        any_agent = env.possible_agents[0]
        obs_dim = env.observation_spaces[any_agent].shape[0]
        act_dim = env.action_spaces[any_agent].shape[0]

    agent = PPOShared(obs_dim, act_dim)
    agent.load(args.ckpt, map_location="cpu")

    if args.mode == "single":
        obs, _ = env.reset()
        while True:
            a, _, _ = agent.act(obs)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                obs, _ = env.reset()
    else:
        obs, _ = env.reset()
        while True:
            acts = {}
            for aname, ob in obs.items():
                a, _, _ = agent.act(ob)
                acts[aname] = a
            obs, r, term, trunc, info = env.step(acts)
            # no rendering implemented for swarm (kept lightweight)
            if len(obs) == 0:
                obs, _ = env.reset()

if __name__ == "__main__":
    main()
