import os, json, time
from dataclasses import asdict

class JSONLogger:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "metrics.jsonl")
        self._t0 = time.time()

    def log(self, **kwargs):
        kwargs["_time"] = time.time() - self._t0
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kwargs) + "\n")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
