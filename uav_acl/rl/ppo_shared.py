import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import ActorCritic
from .buffer import RolloutBuffer

def gaussian_log_prob(mu, log_std, act):
    std = torch.exp(log_std)
    var = std.pow(2)
    logp = -0.5 * (((act - mu) ** 2) / (var + 1e-8) + 2 * log_std + np.log(2*np.pi))
    return logp.sum(dim=-1)

def gaussian_sample(mu, log_std):
    std = torch.exp(log_std)
    eps = torch.randn_like(mu)
    return mu + eps * std

class PPOShared:
    """Parameter-sharing PPO that works for:
    - single-agent Gym env: obs shape (obs_dim,)
    - multi-agent PettingZoo ParallelEnv: dict obs per agent
      We treat each agent at each step as a sample for the shared policy.
    """
    def __init__(self, obs_dim, act_dim, device="cpu",
                 hidden=128, lr=3e-4, clip=0.2, vf_coef=0.5, ent_coef=0.01,
                 gamma=0.99, lam=0.95, train_epochs=5, batch_size=256, rollout_size=4096):
        self.device = torch.device(device)
        self.ac = ActorCritic(obs_dim, act_dim, hidden=hidden).to(self.device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)

        self.clip = clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.lam = lam
        self.train_epochs = train_epochs
        self.batch_size = batch_size

        self.buf = RolloutBuffer(obs_dim, act_dim, rollout_size, self.device)

    @torch.no_grad()
    def act(self, obs_np):
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        mu, log_std, v = self.ac(obs)
        a = gaussian_sample(mu, log_std)
        logp = gaussian_log_prob(mu, log_std, a)
        return a.squeeze(0).cpu().numpy(), float(logp.squeeze(0).cpu().numpy()), float(v.squeeze(0).cpu().numpy())

    def update(self, last_val=0.0):
        self.buf.compute_gae(last_val, gamma=self.gamma, lam=self.lam)

        for _ in range(self.train_epochs):
            for obs, act, logp_old, adv, ret in self.buf.get(self.batch_size):
                mu, log_std, v = self.ac(obs)
                logp = gaussian_log_prob(mu, log_std, act)
                ratio = torch.exp(logp - logp_old)

                # clipped policy loss
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                pi_loss = -torch.min(surr1, surr2).mean()

                # value loss
                v_loss = ((ret - v) ** 2).mean()

                # entropy
                ent = (0.5 + 0.5*np.log(2*np.pi) + log_std).sum(dim=-1).mean()

                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()

        self.buf.reset()

    def save(self, path):
        torch.save({"state_dict": self.ac.state_dict()}, path)

    def load(self, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        self.ac.load_state_dict(ckpt["state_dict"])
