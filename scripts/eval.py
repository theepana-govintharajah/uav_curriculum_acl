import argparse, os, glob
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    args = ap.parse_args()

    rows = []
    for run in sorted(glob.glob(os.path.join(args.runs_dir, "*_seed*"))):
        prog = os.path.join(run, "progress.csv")
        if not os.path.exists(prog):
            continue
        df = pd.read_csv(prog)
        last = df.iloc[-1]
        rows.append({
            "run": os.path.basename(run),
            "steps": int(last["steps"]),
            "stage": int(last["stage"]),
            "train_sr": float(last["train_sr"]),
            "eval_sr": float(last["eval_sr"]),
            "eval_return": float(last["eval_return"]),
        })
    if not rows:
        print("No runs found.")
        return
    out = pd.DataFrame(rows).sort_values(["eval_sr","eval_return"], ascending=False)
    out_path = os.path.join(args.runs_dir, "summary.csv")
    out.to_csv(out_path, index=False)
    print(out)
    print(f"Saved summary to: {out_path}")

if __name__ == "__main__":
    main()
