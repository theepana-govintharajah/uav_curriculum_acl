import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .common import rand_pos, wrap01, dist, clip_norm

class SingleUAVNavEnv(gym.Env):
    """Step 1 environment: Single UAV navigation to a goal.

    Continuous 2D point-mass dynamics with drag:
        v_{t+1} = (1-drag)*v_t + (a_t + wind)*dt
        p_{t+1} = p_t + v_{t+1}*dt

    Observations:
        [p_x, p_y, v_x, v_y, goal_dx, goal_dy, wind_x, wind_y]
    Action:
        2D acceleration in [-1, 1] scaled by max_acc.

    Rewards:
        - small step penalty
        - shaped negative distance to goal
        + goal_reward on reaching goal
        collision penalty if hits obstacles (optional in later configs)
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, cfg, render_mode=None, seed=0):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.world = float(cfg.world_size)
        self.dt = float(cfg.dt)
        self.max_steps = int(cfg.max_steps)

        self.drag = 0.08
        self.max_acc = 2.0

        obs_dim = 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # obstacles for later curriculum steps (still supported here)
        self.n_obs = int(cfg.n_obstacles)
        self.obs_r = float(cfg.obstacle_radius)
        self.dynamic_obstacles = bool(cfg.dynamic_obstacles)
        self.obs_speed = float(cfg.obstacle_speed)

        # wind
        self.wind_enabled = bool(cfg.wind_enabled)
        self.wind_strength = float(cfg.wind_strength)
        self.wind_change_prob = float(cfg.wind_change_prob)
        self.wind = np.zeros(2, dtype=np.float32)

        # energy (disabled for step 1 by default)
        self.energy_enabled = bool(cfg.energy_enabled)
        self.energy_budget = float(cfg.energy_budget)
        self.energy = self.energy_budget
        self.e_cost_move = float(cfg.energy_cost_move)
        self.e_cost_drag = float(cfg.energy_cost_drag)

        self.goal_reward = float(cfg.goal_reward)
        self.goal_radius = float(cfg.goal_radius)

        self._viewer = None
        self._pygame = None
        self._screen = None
        self._clock = None
        self._scale = 60  # pixels per meter-ish

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
        # bounce off walls
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

    def _obs(self):
        gvec = (self.goal - self.p).astype(np.float32)
        return np.array([self.p[0], self.p[1], self.v[0], self.v[1], gvec[0], gvec[1], self.wind[0], self.wind[1]],
                        dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.p = rand_pos(self.rng, self.world, margin=1.0)
        self.v = self.rng.normal(0, 0.2, size=(2,)).astype(np.float32)
        self.goal = rand_pos(self.rng, self.world, margin=1.0)
        self._sample_obstacles()
        self.wind[:] = 0.0
        self._maybe_update_wind()
        self.energy = self.energy_budget
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        a = np.array(action, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0) * self.max_acc
        a = clip_norm(a, self.max_acc)

        self._maybe_update_wind()
        self._update_obstacles()

        # dynamics
        self.v = (1.0 - self.drag) * self.v + (a + self.wind) * self.dt
        self.p = wrap01(self.p + self.v * self.dt, self.world)

        # energy
        if self.energy_enabled:
            self.energy -= self.e_cost_move * float(np.linalg.norm(a))
            self.energy -= self.e_cost_drag * float(np.linalg.norm(self.v))
            if self.energy <= 0.0:
                terminated = True
                reward = float(self.cfg.energy_penalty_empty)
                return self._obs(), reward, terminated, False, {"energy_depleted": True}

        # reward shaping
        d = dist(self.p, self.goal)
        reward = -0.01 - 0.05 * d

        # collision
        if self._collision(self.p):
            reward -= 5.0
            terminated = True
        else:
            terminated = False

        # goal
        if d <= self.goal_radius:
            reward += self.goal_reward
            terminated = True

        truncated = (self.t >= self.max_steps)
        if self.render_mode == "human":
            self.render()
        info = {"distance": d, "energy": float(self.energy)}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._pygame is None:
            import pygame
            self._pygame = pygame
            pygame.init()
            w = int(self.world * self._scale)
            self._screen = pygame.display.set_mode((w, w))
            self._clock = pygame.time.Clock()

        pygame = self._pygame
        self._screen.fill((245, 245, 245))
        wpx = int(self.world * self._scale)

        # draw goal
        gx, gy = (self.goal * self._scale).astype(int)
        pygame.draw.circle(self._screen, (0, 160, 0), (gx, gy), int(self.goal_radius*self._scale), 2)

        # obstacles
        for i in range(self.n_obs):
            ox, oy = (self.obstacles[i] * self._scale).astype(int)
            pygame.draw.circle(self._screen, (180, 50, 50), (ox, oy), int(self.obs_r*self._scale))

        # UAV
        px, py = (self.p * self._scale).astype(int)
        pygame.draw.circle(self._screen, (20, 70, 160), (px, py), 6)

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
