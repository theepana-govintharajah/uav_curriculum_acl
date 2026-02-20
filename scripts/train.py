import argparse, os
from uav_acl.config import EnvConfig
from uav_acl.curriculum import make_schedule
from uav_acl.train_loops import train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", type=str, default="resource_aware", choices=["resource_aware","none","linear","random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    base = EnvConfig(
        world_size=10.0,
        max_steps=260,
        # keep these as defaults; curriculum will override per stage
        n_agents=4,
        n_obstacles=6,
        n_tasks=6,
        wind_enabled=True,
        wind_strength=0.4,

        # energy/comms initial (will be adapted)
        energy_enabled=True,
        energy_budget=60.0,
        comm_enabled=True,
        comm_range=6.0,
        packet_drop_prob=0.05,
        comm_delay_steps=0,
    )

    out_dir = os.path.join("runs", f"{args.schedule}_seed{args.seed}")
    sched = make_schedule(args.schedule, base, total_iters=120, seed=args.seed)

    train(schedule=sched, base_cfg=base, out_dir=out_dir, mode="swarm", seed=args.seed,
          total_env_steps=args.steps, device=args.device)

    print(f"Done. Results in: {out_dir}")

if __name__ == "__main__":
    main()
