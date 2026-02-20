import copy
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class CurriculumState:
    stage: int = 1
    ema_success: float = 0.0
    ema_alpha: float = 0.05

class BaseSchedule:
    def __init__(self, base_cfg):
        self.base_cfg = copy.deepcopy(base_cfg)
        self.state = CurriculumState()

    def reset(self):
        self.state = CurriculumState()

    def update(self, success_rate: float):
        # EMA tracking
        a = self.state.ema_alpha
        self.state.ema_success = (1-a)*self.state.ema_success + a*success_rate

    def get_cfg(self):
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__

class NoneSchedule(BaseSchedule):
    """No curriculum: train directly on the final hardest setting."""
    def get_cfg(self):
        cfg = copy.deepcopy(self.base_cfg)
        # force hardest:
        cfg.n_agents = max(cfg.n_agents, 4)
        cfg.dynamic_obstacles = True
        cfg.n_obstacles = max(cfg.n_obstacles, 6)
        cfg.n_tasks = max(cfg.n_tasks, 6)
        cfg.dynamic_task_arrival = True
        cfg.energy_enabled = True
        cfg.energy_budget = min(cfg.energy_budget, 25.0)
        cfg.comm_enabled = True
        cfg.comm_range = min(cfg.comm_range, 3.5)
        cfg.packet_drop_prob = max(cfg.packet_drop_prob, 0.15)
        cfg.wind_enabled = True
        cfg.wind_strength = max(cfg.wind_strength, 0.6)
        return cfg

class LinearSchedule(BaseSchedule):
    """Linear: advance stages by training progress fraction."""
    def __init__(self, base_cfg, total_iters: int):
        super().__init__(base_cfg)
        self.total_iters = max(1, int(total_iters))
        self.iter = 0

    def step_iter(self):
        self.iter += 1

    def get_cfg(self):
        frac = min(1.0, self.iter / self.total_iters)
        # map frac to stage 1..6
        stage = 1 + int(frac * 5.999)
        self.state.stage = stage
        return build_stage_cfg(self.base_cfg, stage)

class RandomSchedule(BaseSchedule):
    """Randomly samples a stage each time (bad baseline)."""
    def __init__(self, base_cfg, seed: int = 0):
        super().__init__(base_cfg)
        self.rng = np.random.default_rng(seed)

    def get_cfg(self):
        stage = int(self.rng.integers(1, 7))
        self.state.stage = stage
        return build_stage_cfg(self.base_cfg, stage)

class ResourceAwareSchedule(BaseSchedule):
    """Adaptive schedule inspired by success-rate band control.

    - First follow your 1..4 structural stages (capability building).
    - Then *adaptively* tightens energy + communication to keep SR in [srl, srh].
    """
    def __init__(self, base_cfg, srl=0.45, srh=0.75, delta=0.05):
        super().__init__(base_cfg)
        self.srl = float(srl)
        self.srh = float(srh)
        self.delta = float(delta)

        # parameters we adapt after stage>=4
        self.energy_budget = float(base_cfg.energy_budget)
        self.comm_range = float(base_cfg.comm_range)
        self.packet_drop = float(base_cfg.packet_drop_prob)

    def get_cfg(self):
        stage = self.state.stage
        cfg = build_stage_cfg(self.base_cfg, stage)

        # after stage 4, adapt energy + comm automatically
        if stage >= 4:
            sr = self.state.ema_success
            # If too successful => make harder (tighten budgets / reduce comm)
            if sr > self.srh:
                self.energy_budget = max(12.0, self.energy_budget * (1.0 - self.delta))
                self.comm_range = max(2.5, self.comm_range * (1.0 - self.delta))
                self.packet_drop = min(0.35, self.packet_drop + 0.02)
            # If failing => make easier
            elif sr < self.srl:
                self.energy_budget = min(80.0, self.energy_budget * (1.0 + self.delta))
                self.comm_range = min(9999.0, self.comm_range * (1.0 + self.delta))
                self.packet_drop = max(0.0, self.packet_drop - 0.02)

            cfg.energy_enabled = True
            cfg.energy_budget = float(self.energy_budget)
            cfg.comm_enabled = True
            cfg.comm_range = float(self.comm_range)
            cfg.packet_drop_prob = float(self.packet_drop)

        return cfg

    def maybe_advance_stage(self):
        # advance if EMA stable above threshold, up to 6
        if self.state.stage < 6 and self.state.ema_success > 0.68:
            self.state.stage += 1

def build_stage_cfg(base_cfg, stage: int):
    import copy
    cfg = copy.deepcopy(base_cfg)

    # Stage mapping (your exact order)
    if stage == 1:
        # single UAV navigation, unlimited energy
        cfg.n_agents = 1
        cfg.n_obstacles = 0
        cfg.dynamic_obstacles = False
        cfg.n_tasks = 0
        cfg.dynamic_task_arrival = False
        cfg.energy_enabled = False
        cfg.comm_enabled = False
        cfg.wind_enabled = False

    elif stage == 2:
        # UAV swarms with dynamic obstacles
        cfg.n_agents = max(cfg.n_agents, 3)
        cfg.n_obstacles = max(cfg.n_obstacles, 6)
        cfg.dynamic_obstacles = True
        cfg.n_tasks = 0
        cfg.dynamic_task_arrival = False
        cfg.energy_enabled = False
        cfg.comm_enabled = False
        cfg.wind_enabled = True
        cfg.wind_strength = max(cfg.wind_strength, 0.3)

    elif stage == 3:
        # cooperative task allocation (static tasks)
        cfg.n_agents = max(cfg.n_agents, 4)
        cfg.n_obstacles = max(cfg.n_obstacles, 6)
        cfg.dynamic_obstacles = True
        cfg.n_tasks = max(cfg.n_tasks, 6)
        cfg.dynamic_task_arrival = False
        cfg.energy_enabled = False
        cfg.comm_enabled = False
        cfg.wind_enabled = True
        cfg.wind_strength = max(cfg.wind_strength, 0.4)

    elif stage == 4:
        # dynamic task arrival
        cfg.n_agents = max(cfg.n_agents, 4)
        cfg.n_obstacles = max(cfg.n_obstacles, 6)
        cfg.dynamic_obstacles = True
        cfg.n_tasks = max(cfg.n_tasks, 4)
        cfg.dynamic_task_arrival = True
        cfg.task_arrival_prob = max(cfg.task_arrival_prob, 0.04)
        cfg.energy_enabled = False
        cfg.comm_enabled = False
        cfg.wind_enabled = True
        cfg.wind_strength = max(cfg.wind_strength, 0.5)

    elif stage == 5:
        # energy budget tightness
        cfg.n_agents = max(cfg.n_agents, 4)
        cfg.n_obstacles = max(cfg.n_obstacles, 6)
        cfg.dynamic_obstacles = True
        cfg.n_tasks = max(cfg.n_tasks, 4)
        cfg.dynamic_task_arrival = True
        cfg.energy_enabled = True
        cfg.energy_budget = min(cfg.energy_budget, 35.0)
        cfg.comm_enabled = False
        cfg.wind_enabled = True
        cfg.wind_strength = max(cfg.wind_strength, 0.55)

    elif stage == 6:
        # communication sparsity + energy tightness (final)
        cfg.n_agents = max(cfg.n_agents, 4)
        cfg.n_obstacles = max(cfg.n_obstacles, 6)
        cfg.dynamic_obstacles = True
        cfg.n_tasks = max(cfg.n_tasks, 4)
        cfg.dynamic_task_arrival = True
        cfg.energy_enabled = True
        cfg.energy_budget = min(cfg.energy_budget, 25.0)
        cfg.comm_enabled = True
        cfg.comm_range = min(cfg.comm_range, 3.5)
        cfg.packet_drop_prob = max(cfg.packet_drop_prob, 0.15)
        cfg.comm_delay_steps = max(cfg.comm_delay_steps, 1)
        cfg.wind_enabled = True
        cfg.wind_strength = max(cfg.wind_strength, 0.6)

    else:
        raise ValueError("stage must be 1..6")

    return cfg

def make_schedule(name: str, base_cfg, total_iters: int = 200, seed: int = 0):
    name = name.lower().strip()
    if name in ["none", "no", "nocurriculum"]:
        return NoneSchedule(base_cfg)
    if name in ["linear"]:
        return LinearSchedule(base_cfg, total_iters=total_iters)
    if name in ["random"]:
        return RandomSchedule(base_cfg, seed=seed)
    if name in ["resource_aware", "resource-aware", "adaptive"]:
        return ResourceAwareSchedule(base_cfg)
    raise ValueError(f"Unknown schedule: {name}")
