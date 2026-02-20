import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.v = nn.Linear(hidden, 1)

    def forward(self, obs):
        x = F.tanh(self.fc1(obs))
        x = F.tanh(self.fc2(x))
        mu = self.mu(x)
        v = self.v(x).squeeze(-1)
        log_std = self.log_std.expand_as(mu)
        return mu, log_std, v
