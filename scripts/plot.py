import argparse, os, glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    args = ap.parse_args()

    runs = sorted(glob.glob(os.path.join(args.runs_dir, "*_seed*")))
    if not runs:
        print("No runs found.")
        return

    plt.figure()
    for run in runs:
        prog = os.path.join(run, "progress.csv")
        if not os.path.exists(prog):
            continue
        df = pd.read_csv(prog)
        label = os.path.basename(run)
        plt.plot(df["steps"], df["eval_sr"], label=label)

    plt.xlabel("Environment steps")
    plt.ylabel("Eval success rate")
    plt.legend()
    out_path = os.path.join(args.runs_dir, "compare_eval_sr.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_path}")

    plt.figure()
    for run in runs:
        prog = os.path.join(run, "progress.csv")
        if not os.path.exists(prog):
            continue
        df = pd.read_csv(prog)
        label = os.path.basename(run)
        plt.plot(df["steps"], df["eval_return"], label=label)

    plt.xlabel("Environment steps")
    plt.ylabel("Eval return")
    plt.legend()
    out_path2 = os.path.join(args.runs_dir, "compare_eval_return.png")
    plt.savefig(out_path2, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_path2}")

if __name__ == "__main__":
    main()
