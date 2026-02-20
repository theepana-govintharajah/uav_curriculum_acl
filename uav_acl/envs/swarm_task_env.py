import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
from .common import rand_pos, wrap01, dist, clip_norm

class UAVSwarmTaskEnv(ParallelEnv):
    """Multi-agent continuous swarm environment (PettingZoo ParallelEnv).

    Agents move in 2D continuous space with drag + wind.
    Tasks are points that can be completed by getting within task_radius.

    Curriculum steps supported via EnvConfig toggles:
    - dynamic obstacles
    - cooperative task allocation (multi agents)
    - dynamic task arrival
    - energy budget tightness
    - communication sparsity (comm range + packet drop + delay)

    Observation per-agent (fixed size, padded):
        own: p(2), v(2), energy(1)
        goal/task summary: vector to nearest task (2) + num_active_tasks(1)
        neighbors: up to K nearest neighbors within comm_range:
            rel_pos(2) + rel_vel(2) for each neighbor (padded with zeros)
        wind(2)
    Action:
        acceleration (2) in [-1,1] scaled by max_acc.

    Reward:
        shared cooperative reward for completing tasks (optional)
        shaping: negative distance to nearest task
        step penalty
        collision penalty
        energy depletion penalty
    """

    metadata = {"name": "uav_swarm_task_v0"}

    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.world = float(cfg.world_size)
        self.dt = float(cfg.dt)
        self.max_steps = int(cfg.max_steps)
        self.n_agents = int(cfg.n_agents)
        self.agents = [f"uav_{i}" for i in range(self.n_agents)]
        self.possible_agents = list(self.agents)

        self.drag = 0.08
        self.max_acc = 2.0

        # neighbors to include
        self.K = 3  # fixed padding

        # spaces
        obs_dim = 2+2+1 + 2+1 + self.K*(2+2) + 2
        self._obs_dim = obs_dim
        self.observation_spaces = {a: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                                   for a in self.agents}
        self.action_spaces = {a: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
                              for a in self.agents}

        # obstacles
        self.n_obs = int(cfg.n_obstacles)
        self.obs_r = float(cfg.obstacle_radius)
        self.dynamic_obstacles = bool(cfg.dynamic_obstacles)
        self.obs_speed = float(cfg.obstacle_speed)

        # tasks
        self.n_tasks_init = int(cfg.n_tasks)
        self.task_r = float(cfg.task_radius)
        self.task_reward = float(cfg.task_reward)
        self.coop_reward = bool(cfg.cooperative_reward)
        self.dynamic_task_arrival = bool(cfg.dynamic_task_arrival)
        self.task_arrival_prob = float(cfg.task_arrival_prob)
        self.max_active_tasks = int(cfg.max_active_tasks)

        # energy
        self.energy_enabled = bool(cfg.energy_enabled)
        self.energy_budget = float(cfg.energy_budget)
        self.e_cost_move = float(cfg.energy_cost_move)
        self.e_cost_drag = float(cfg.energy_cost_drag)
        self.energy_penalty_empty = float(cfg.energy_penalty_empty)

        # wind
        self.wind_enabled = bool(cfg.wind_enabled)
        self.wind_strength = float(cfg.wind_strength)
        self.wind_change_prob = float(cfg.wind_change_prob)
        self.wind = np.zeros(2, dtype=np.float32)

        # comm
        self.comm_enabled = bool(cfg.comm_enabled)
        self.comm_range = float(cfg.comm_range)
        self.packet_drop_prob = float(cfg.packet_drop_prob)
        self.comm_delay_steps = int(cfg.comm_delay_steps)
        self._msg_history = None  # for delay simulation

        self.reset()

    def _sample_obstacles(self):
        self.obstacles = np.zeros((self.n_obs, 2), dtype=np.float32)
        self.obst_vel = np.zeros((self.n_obs, 2), dtype=np.float32)
        for i in range(self.n_obs):
            self.obstacles[i] = rand_pos(self.rng, self.world, margin=1.0)
            v = self.rng.normal(0, 1, size=(2,)).astype(np.float32)
            n = np.linalg.norm(v) + 1e-8
            self.obst_vel[i] = (v / n) * self.obs_speed

    def _maybe_update_wind(self):
        if not self.wind_enabled:
            self.wind[:] = 0.0
            return
        if self.rng.random() < self.wind_change_prob:
            ang = self.rng.uniform(0, 2*np.pi)
            mag = self.rng.uniform(0, self.wind_strength)
            self.wind[:] = np.array([np.cos(ang)*mag, np.sin(ang)*mag], dtype=np.float32)

    def _update_obstacles(self):
        if self.n_obs <= 0:
            return
        if not self.dynamic_obstacles:
            return
        self.obstacles = wrap01(self.obstacles + self.obst_vel * self.dt, self.world)
        for i in range(self.n_obs):
            for d in range(2):
                if self.obstacles[i, d] <= 0.0 or self.obstacles[i, d] >= self.world:
                    self.obst_vel[i, d] *= -1.0

    def _collision(self, p):
        if self.n_obs <= 0:
            return False
        for i in range(self.n_obs):
            if dist(p, self.obstacles[i]) <= self.obs_r:
                return True
        return False

    def _spawn_tasks(self, n):
        tasks = []
        for _ in range(n):
            tasks.append(rand_pos(self.rng, self.world, margin=1.0))
        return tasks

    def _active_tasks_arrival(self):
        if not self.dynamic_task_arrival:
            return
        if len(self.tasks) >= self.max_active_tasks:
            return
        if self.rng.random() < self.task_arrival_prob:
            self.tasks.extend(self._spawn_tasks(1))

    def _nearest_task_vec(self, p):
        if len(self.tasks) == 0:
            return np.zeros(2, dtype=np.float32), 0.0
        dists = np.linalg.norm(np.stack(self.tasks, axis=0) - p[None, :], axis=1)
        j = int(np.argmin(dists))
        vec = (self.tasks[j] - p).astype(np.float32)
        return vec, float(dists[j])

    def _neighbors_features(self, i):
        # return K neighbors rel_pos, rel_vel, padded. Apply comm constraints.
        pi = self.P[i]
        vi = self.V[i]
        rels = []
        for j in range(self.n_agents):
            if j == i: 
                continue
            pj = self.P[j]
            vj = self.V[j]
            d = float(np.linalg.norm(pj - pi))
            if self.comm_enabled:
                if d > self.comm_range:
                    continue
                if self.rng.random() < self.packet_drop_prob:
                    continue
            rels.append((d, (pj - pi).astype(np.float32), (vj - vi).astype(np.float32)))
        rels.sort(key=lambda x: x[0])
        feat = []
        for k in range(self.K):
            if k < len(rels):
                feat.append(rels[k][1])
                feat.append(rels[k][2])
            else:
                feat.append(np.zeros(2, dtype=np.float32))
                feat.append(np.zeros(2, dtype=np.float32))
        return np.concatenate(feat, axis=0).astype(np.float32)

    def _obs_agent(self, i):
        p = self.P[i].astype(np.float32)
        v = self.V[i].astype(np.float32)
        e = np.array([self.E[i]], dtype=np.float32)

        tvec, _ = self._nearest_task_vec(p)
        tcount = np.array([float(len(self.tasks))], dtype=np.float32)

        nfeat = self._neighbors_features(i)
        w = self.wind.astype(np.float32)

        obs = np.concatenate([p, v, e, tvec, tcount, nfeat, w], axis=0)
        assert obs.shape[0] == self._obs_dim
        return obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.t = 0

        self.P = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.V = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.E = np.full((self.n_agents,), self.energy_budget, dtype=np.float32)

        for i in range(self.n_agents):
            self.P[i] = rand_pos(self.rng, self.world, margin=1.0)
            self.V[i] = self.rng.normal(0, 0.2, size=(2,)).astype(np.float32)

        self._sample_obstacles()
        self.wind[:] = 0.0
        self._maybe_update_wind()

        self.tasks = self._spawn_tasks(self.n_tasks_init)

        # comm delay: store history of positions/vels for delayed reads
        if self.comm_enabled and self.comm_delay_steps > 0:
            self._msg_history = [(self.P.copy(), self.V.copy())]
        else:
            self._msg_history = None

        obs = {a: self._obs_agent(i) for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def _apply_comm_delay(self):
        # If delay enabled, neighbors features should be based on delayed states.
        if self._msg_history is None:
            return
        self._msg_history.append((self.P.copy(), self.V.copy()))
        if len(self._msg_history) > self.comm_delay_steps + 1:
            self._msg_history.pop(0)
        delayed_P, delayed_V = self._msg_history[0]
        return delayed_P, delayed_V

    def step(self, actions):
        self.t += 1
        self._maybe_update_wind()
        self._update_obstacles()
        self._active_tasks_arrival()

        # Optionally use delayed states for comm features, but dynamics use true states.
        delayed = None
        if self.comm_enabled and self.comm_delay_steps > 0:
            delayed = self._apply_comm_delay()

        # dynamics update per agent
        terminated = {a: False for a in self.agents}
        truncated = {a: False for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        infos = {a: {} for a in self.agents}

        # apply actions
        for i, a_name in enumerate(self.agents):
            a = np.array(actions[a_name], dtype=np.float32)
            a = np.clip(a, -1.0, 1.0) * self.max_acc
            a = clip_norm(a, self.max_acc)

            self.V[i] = (1.0 - self.drag) * self.V[i] + (a + self.wind) * self.dt
            self.P[i] = wrap01(self.P[i] + self.V[i] * self.dt, self.world)

            # energy
            if self.energy_enabled:
                self.E[i] -= self.e_cost_move * float(np.linalg.norm(a))
                self.E[i] -= self.e_cost_drag * float(np.linalg.norm(self.V[i]))
                if self.E[i] <= 0.0:
                    terminated[a_name] = True
                    rewards[a_name] += self.energy_penalty_empty
                    infos[a_name]["energy_depleted"] = True

        # collision penalties
        for i, a_name in enumerate(self.agents):
            if terminated[a_name]:
                continue
            if self._collision(self.P[i]):
                terminated[a_name] = True
                rewards[a_name] -= 5.0
                infos[a_name]["collision"] = True

        # task completion (any agent can complete any task)
        completed = []
        for ti, tp in enumerate(self.tasks):
            for i in range(self.n_agents):
                if dist(self.P[i], tp) <= self.task_r:
                    completed.append(ti)
                    break
        if completed:
            completed = sorted(set(completed), reverse=True)
            for ti in completed:
                self.tasks.pop(ti)
            # cooperative reward: shared team reward for each completed task
            r_add = self.task_reward * float(len(completed))
            if self.coop_reward:
                for a_name in self.agents:
                    if not terminated[a_name]:
                        rewards[a_name] += r_add
            else:
                # local reward: give to all non-terminated anyway (kept simple)
                for a_name in self.agents:
                    if not terminated[a_name]:
                        rewards[a_name] += r_add

        # shaping: negative distance to nearest task
        for i, a_name in enumerate(self.agents):
            if terminated[a_name]:
                continue
            _, dmin = self._nearest_task_vec(self.P[i])
            rewards[a_name] += (-0.01 - 0.03 * dmin)

        # termination if time
        done_time = (self.t >= self.max_steps)
        if done_time:
            for a_name in self.agents:
                truncated[a_name] = True

        # episode ends for an agent if terminated or truncated; in ParallelEnv, agents list can shrink
        # keep agents present but mark done; PettingZoo expects removed agents, but ParallelEnv allows fixed set if we follow API:
        # We'll remove done agents for correctness.
        dead = [a for a in self.agents if terminated[a] or truncated[a]]
        for a in dead:
            if a in self.agents:
                self.agents.remove(a)

        # build observations for remaining agents
        obs = {}
        for i, a_name in enumerate(self.possible_agents):
            if a_name not in self.agents:
                continue
            idx = int(a_name.split('_')[1])
            if delayed is not None:
                # temporarily swap in delayed states for neighbor encoding
                P_saved, V_saved = self.P, self.V
                self.P, self.V = delayed
                obs[a_name] = self._obs_agent(idx)
                self.P, self.V = P_saved, V_saved
            else:
                obs[a_name] = self._obs_agent(idx)

        # Add info
        for i_name in list(rewards.keys()):
            infos[i_name]["n_tasks"] = len(self.tasks)

        return obs, rewards, terminated, truncated, infos
