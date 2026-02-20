import os
import numpy as np
import torch

from .envs import SingleUAVNavEnv, UAVSwarmTaskEnv
from .rl import PPOShared

def make_env(cfg, mode: str, seed: int, render: bool=False):
    if mode == "single":
        return SingleUAVNavEnv(cfg, render_mode="human" if render else None, seed=seed)
    if mode == "swarm":
        return UAVSwarmTaskEnv(cfg, seed=seed)
    raise ValueError("mode must be 'single' or 'swarm'")

def episode_success_from_info(info: dict) -> bool:
    # Single-agent: success if distance <= goal_radius at termination
    if "distance" in info:
        return info["distance"] <= 0.8
    return False

def train(schedule, base_cfg, out_dir, mode="swarm", seed=0, total_env_steps=200_000,
          rollout_size=4096, eval_every=10_000, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # create initial env to get dims
    cfg = schedule.get_cfg()
    env = make_env(cfg, mode, seed=seed, render=False)

    if mode == "single":
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    else:
        # all agents same dims
        any_agent = env.possible_agents[0]
        obs_dim = env.observation_spaces[any_agent].shape[0]
        act_dim = env.action_spaces[any_agent].shape[0]

    agent = PPOShared(obs_dim, act_dim, device=device, rollout_size=rollout_size)

    # training state
    steps = 0
    ep_returns = []
    ep_success = []
    ep_len = []

    # initial reset
    if mode == "single":
        obs, _ = env.reset(seed=seed)
        active = True
        cur_ret = 0.0
        cur_len = 0
    else:
        obs, _ = env.reset(seed=seed)
        cur_ret = {a: 0.0 for a in env.possible_agents}
        cur_len = {a: 0 for a in env.possible_agents}

    # helper for evaluation
    def evaluate(eval_episodes=10):
        nonlocal env
        cfg_eval = schedule.get_cfg()
        env_eval = make_env(cfg_eval, mode, seed=seed+999, render=False)
        succ = 0
        rew = []
        for _ in range(eval_episodes):
            if mode == "single":
                o, _ = env_eval.reset()
                rsum = 0.0
                for _t in range(cfg_eval.max_steps):
                    a, _, _ = agent.act(o)
                    o, r, term, trunc, info = env_eval.step(a)
                    rsum += r
                    if term or trunc:
                        if info.get("distance", 999) <= cfg_eval.goal_radius:
                            succ += 1
                        break
                rew.append(rsum)
            else:
                o, _ = env_eval.reset()
                rsum = 0.0
                # run until all agents done
                while True:
                    acts = {}
                    for aname, ob in o.items():
                        a, _, _ = agent.act(ob)
                        acts[aname] = a
                    o, r, term, trunc, info = env_eval.step(acts)
                    rsum += sum(r.values()) if isinstance(r, dict) else float(r)
                    if len(o) == 0:
                        # success proxy: finished many tasks
                        # define success if tasks remaining small
                        # info dict still contains keys for all agents in last step, but we track from any
                        any_info = next(iter(info.values())) if isinstance(info, dict) and len(info)>0 else {}
                        if any_info.get("n_tasks", 999) <= 1:
                            succ += 1
                        break
                rew.append(rsum)
        return succ / eval_episodes, float(np.mean(rew))

    # main training loop
    last_eval = 0
    while steps < total_env_steps:
        # update env cfg if schedule changes (for curriculum)
        cfg = schedule.get_cfg()

        # If mode mismatch with stage 1 (single) vs others, auto switch:
        # Stage 1 should be single agent; stage>=2 should be swarm.
        # We'll follow your step order automatically.
        if hasattr(schedule, "state") and schedule.state.stage == 1:
            if mode != "single":
                mode = "single"
        else:
            if mode != "swarm":
                mode = "swarm"

        # recreate env if config changed meaningfully (simple: recreate each eval block)
        # To keep code stable: recreate every time we enter a new stage.
        # Detect stage transitions if schedule has state.stage
        # We'll use schedule.state.stage and store last_stage.
        if not hasattr(train, "_last_stage"):
            train._last_stage = None
        stage = getattr(schedule.state, "stage", None)
        if stage != train._last_stage:
            # close old env
            try:
                env.close()
            except Exception:
                pass
            env = make_env(cfg, mode, seed=seed + steps//1000, render=False)
            if mode == "single":
                obs, _ = env.reset(seed=seed + steps//1000)
                cur_ret = 0.0
                cur_len = 0
            else:
                obs, _ = env.reset(seed=seed + steps//1000)
                cur_ret = {a: 0.0 for a in env.possible_agents}
                cur_len = {a: 0 for a in env.possible_agents}
            train._last_stage = stage

        # collect rollouts
        while not agent.buf.full() and steps < total_env_steps:
            if mode == "single":
                a, logp, v = agent.act(obs)
                next_obs, r, term, trunc, info = env.step(a)
                done = float(term or trunc)
                agent.buf.add(obs, a, logp, r, done, v)

                cur_ret += r
                cur_len += 1
                steps += 1

                obs = next_obs
                if term or trunc:
                    ep_returns.append(cur_ret)
                    ep_len.append(cur_len)
                    ep_success.append(1.0 if info.get("distance", 999) <= cfg.goal_radius else 0.0)
                    obs, _ = env.reset()
                    cur_ret = 0.0
                    cur_len = 0
            else:
                # multi-agent: each alive agent contributes a sample
                acts = {}
                logps = {}
                vals = {}
                for aname, ob in obs.items():
                    a, logp, v = agent.act(ob)
                    acts[aname] = a
                    logps[aname] = logp
                    vals[aname] = v

                next_obs, r, term, trunc, info = env.step(acts)

                # add per agent sample
                for aname, ob in obs.items():
                    done = float(term.get(aname, False) or trunc.get(aname, False))
                    agent.buf.add(ob, acts[aname], logps[aname], float(r.get(aname, 0.0)), done, float(vals[aname]))
                    cur_ret[aname] += float(r.get(aname, 0.0))
                    cur_len[aname] += 1
                    steps += 1
                    if agent.buf.full() or steps >= total_env_steps:
                        break

                obs = next_obs

                if len(obs) == 0:
                    # episode ended for all
                    # success proxy: tasks remaining small from last info
                    any_info = next(iter(info.values())) if isinstance(info, dict) and len(info)>0 else {}
                    success = 1.0 if any_info.get("n_tasks", 999) <= 1 else 0.0
                    ep_success.append(success)
                    ep_returns.append(sum(cur_ret.values()))
                    ep_len.append(int(np.mean(list(cur_len.values()))))
                    obs, _ = env.reset()
                    cur_ret = {a: 0.0 for a in env.possible_agents}
                    cur_len = {a: 0 for a in env.possible_agents}

        # bootstrap last value for GAE
        last_val = 0.0
        if mode == "single":
            _, _, v = agent.act(obs)
            last_val = v
        else:
            # mean value over alive agents
            if len(obs) > 0:
                vs = []
                for aname, ob in obs.items():
                    _, _, v = agent.act(ob)
                    vs.append(v)
                last_val = float(np.mean(vs)) if vs else 0.0

        agent.update(last_val=last_val)

        # eval schedule + update curriculum
        if (steps - last_eval) >= eval_every and len(ep_success) > 10:
            last_eval = steps
            sr_recent = float(np.mean(ep_success[-20:]))
            schedule.update(sr_recent)
            # stage advancement for resource-aware
            if hasattr(schedule, "maybe_advance_stage"):
                schedule.maybe_advance_stage()

            # evaluation runs
            sr_eval, r_eval = evaluate(eval_episodes=8)

            # save checkpoint
            ckpt_path = os.path.join(ckpt_dir, "latest.pt")
            agent.save(ckpt_path)

            # write summary
            with open(os.path.join(out_dir, "progress.csv"), "a", encoding="utf-8") as f:
                if f.tell() == 0:
                    f.write("steps,stage,train_sr,eval_sr,eval_return,avg_ep_return,avg_ep_len\n")
                f.write(f"{steps},{getattr(schedule.state,'stage',-1)},{sr_recent:.3f},{sr_eval:.3f},{r_eval:.2f},"
                        f"{float(np.mean(ep_returns[-20:])):.2f},{float(np.mean(ep_len[-20:])):.1f}\n")

    # final save
    agent.save(os.path.join(ckpt_dir, "final.pt"))
    return out_dir
