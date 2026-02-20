import os
import numpy as np
import json
from .envs import UAVSwarmTaskEnv
from .rl import PPOShared


def make_env(cfg, seed: int):
    # Always use the PettingZoo swarm env to keep observation/action dimensions fixed.
    return UAVSwarmTaskEnv(cfg, seed=seed)


def _cfg_key(cfg) -> str:
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg.__dict__)
    # stable key for equality checks
    return json.dumps(d, sort_keys=True)


def train(schedule, base_cfg, out_dir, mode="swarm", seed=0, total_env_steps=200_000,
          rollout_size=4096, eval_every=10_000, device="cpu"):
    """Train with parameter-sharing PPO on a fixed-dimension swarm env.

    IMPORTANT:
    - We ALWAYS use UAVSwarmTaskEnv (even for stage 1 with n_agents=1).
    - Curriculum changes ONLY env parameters, not observation dimensions.
    """
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Build initial env
    cfg = schedule.get_cfg()
    env = make_env(cfg, seed=seed)

    any_agent = env.possible_agents[0]
    obs_dim = env.observation_spaces[any_agent].shape[0]
    act_dim = env.action_spaces[any_agent].shape[0]
    agent = PPOShared(obs_dim, act_dim, device=device, rollout_size=rollout_size)

    steps = 0
    ep_returns = []
    ep_success = []
    ep_len = []

    obs, _ = env.reset(seed=seed)
    cur_ret = {a: 0.0 for a in env.possible_agents}
    cur_len = {a: 0 for a in env.possible_agents}

    last_eval = 0
    last_cfg_key = _cfg_key(cfg)

    def evaluate(eval_episodes=8):
        cfg_eval = schedule.get_cfg()
        env_eval = make_env(cfg_eval, seed=seed + 999)
        succ = 0
        rews = []
        for _ in range(eval_episodes):
            o, _ = env_eval.reset()
            rsum = 0.0
            while True:
                acts = {}
                for aname, ob in o.items():
                    a, _, _ = agent.act(ob)
                    acts[aname] = a
                o, r, term, trunc, info = env_eval.step(acts)
                rsum += float(sum(r.values())) if isinstance(r, dict) else float(r)

                if len(o) == 0:
                    any_info = next(iter(info.values())) if isinstance(info, dict) and len(info) > 0 else {}
                    if any_info.get("n_tasks", 999) == 0:
                        succ += 1
                    break
            rews.append(rsum)
        return succ / eval_episodes, float(np.mean(rews))

    while steps < total_env_steps:
        # Update cfg from schedule; recreate env if needed (adaptive tightening, stage changes, etc.)
        cfg = schedule.get_cfg()
        cfg_key = _cfg_key(cfg)
        if cfg_key != last_cfg_key:
            try:
                env.close()
            except Exception:
                pass
            env = make_env(cfg, seed=seed + steps // 1000)
            obs, _ = env.reset(seed=seed + steps // 1000)
            cur_ret = {a: 0.0 for a in env.possible_agents}
            cur_len = {a: 0 for a in env.possible_agents}
            last_cfg_key = cfg_key

        # Collect rollouts
        while not agent.buf.full() and steps < total_env_steps:
            acts = {}
            logps = {}
            vals = {}

            for aname, ob in obs.items():
                a, logp, v = agent.act(ob)
                acts[aname] = a
                logps[aname] = logp
                vals[aname] = v

            next_obs, r, term, trunc, info = env.step(acts)

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
                any_info = next(iter(info.values())) if isinstance(info, dict) and len(info) > 0 else {}
                success = 1.0 if any_info.get("n_tasks", 999) == 0 else 0.0
                ep_success.append(success)
                ep_returns.append(sum(cur_ret.values()))
                ep_len.append(int(np.mean(list(cur_len.values()))))

                obs, _ = env.reset()
                cur_ret = {a: 0.0 for a in env.possible_agents}
                cur_len = {a: 0 for a in env.possible_agents}

        # Bootstrap last value (mean over alive agents)
        last_val = 0.0
        if len(obs) > 0:
            vs = []
            for _, ob in obs.items():
                _, _, v = agent.act(ob)
                vs.append(v)
            last_val = float(np.mean(vs)) if vs else 0.0

        agent.update(last_val=last_val)

        # Periodic evaluation + curriculum update
        if (steps - last_eval) >= eval_every and len(ep_success) > 10:
            last_eval = steps
            sr_recent = float(np.mean(ep_success[-20:]))
            schedule.update(sr_recent)
            if hasattr(schedule, "maybe_advance_stage"):
                schedule.maybe_advance_stage()

            sr_eval, r_eval = evaluate(eval_episodes=8)

            agent.save(os.path.join(ckpt_dir, "latest.pt"))

            with open(os.path.join(out_dir, "progress.csv"), "a", encoding="utf-8") as f:
                if f.tell() == 0:
                    f.write("steps,stage,train_sr,eval_sr,eval_return,avg_ep_return,avg_ep_len\n")
                f.write(
                    f"{steps},{getattr(schedule.state,'stage',-1)},{sr_recent:.3f},{sr_eval:.3f},{r_eval:.2f},"
                    f"{float(np.mean(ep_returns[-20:])):.2f},{float(np.mean(ep_len[-20:])):.1f}\n"
                )

    agent.save(os.path.join(ckpt_dir, "final.pt"))
    return out_dir
