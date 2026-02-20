from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class EnvConfig:
    # Common
    world_size: float = 10.0   # square [0, world_size]^2
    dt: float = 0.1
    max_steps: int = 300

    # Obstacles
    n_obstacles: int = 0
    obstacle_radius: float = 0.5
    dynamic_obstacles: bool = False
    obstacle_speed: float = 0.7

    # Tasks
    n_tasks: int = 0
    task_radius: float = 0.6
    task_reward: float = 5.0
    cooperative_reward: bool = True
    dynamic_task_arrival: bool = False
    task_arrival_prob: float = 0.03   # per step
    max_active_tasks: int = 6

    # Energy
    energy_enabled: bool = False
    energy_budget: float = 9999.0
    energy_cost_move: float = 0.03    # cost per unit acceleration norm
    energy_cost_drag: float = 0.01    # cost per unit velocity norm
    energy_penalty_empty: float = -8.0 # if energy depleted

    # Wind
    wind_enabled: bool = False
    wind_strength: float = 0.0         # acceleration bias magnitude
    wind_change_prob: float = 0.05

    # Communication
    comm_enabled: bool = False
    comm_range: float = 9999.0
    packet_drop_prob: float = 0.0
    comm_delay_steps: int = 0

    # Agents
    n_agents: int = 1

    # Goal (single-agent nav)
    goal_reward: float = 10.0
    goal_radius: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
