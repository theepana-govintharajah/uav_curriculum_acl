import numpy as np

def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= max_norm or n < 1e-8:
        return v
    return v * (max_norm / n)

def wrap01(x: np.ndarray, world_size: float) -> np.ndarray:
    # keep within [0, world_size]
    return np.clip(x, 0.0, world_size)

def rand_pos(rng: np.random.Generator, world_size: float, margin: float = 1.0) -> np.ndarray:
    return rng.uniform(margin, world_size - margin, size=(2,)).astype(np.float32)

def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def seed_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
