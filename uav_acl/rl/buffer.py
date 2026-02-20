import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.logp = np.zeros((size,), dtype=np.float32)
        self.rew = np.zeros((size,), dtype=np.float32)
        self.done = np.zeros((size,), dtype=np.float32)
        self.val = np.zeros((size,), dtype=np.float32)

        self.adv = np.zeros((size,), dtype=np.float32)
        self.ret = np.zeros((size,), dtype=np.float32)

        self.ptr = 0
        self.max = size
        self.device = device

    def add(self, obs, act, logp, rew, done, val):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.logp[i] = logp
        self.rew[i] = rew
        self.done[i] = done
        self.val[i] = val
        self.ptr += 1

    def full(self):
        return self.ptr >= self.max

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_val, gamma=0.99, lam=0.95):
        # GAE-Lambda, with done as episode boundary
        adv = 0.0
        for t in reversed(range(self.ptr)):
            next_nonterminal = 1.0 - self.done[t]
            next_val = last_val if t == self.ptr - 1 else self.val[t+1]
            delta = self.rew[t] + gamma * next_val * next_nonterminal - self.val[t]
            adv = delta + gamma * lam * next_nonterminal * adv
            self.adv[t] = adv
        self.ret[:self.ptr] = self.adv[:self.ptr] + self.val[:self.ptr]

        # normalize adv
        a = self.adv[:self.ptr]
        self.adv[:self.ptr] = (a - a.mean()) / (a.std() + 1e-8)

    def get(self, batch_size):
        idx = np.random.permutation(self.ptr)
        for start in range(0, self.ptr, batch_size):
            b = idx[start:start+batch_size]
            yield (
                torch.tensor(self.obs[b], device=self.device),
                torch.tensor(self.act[b], device=self.device),
                torch.tensor(self.logp[b], device=self.device),
                torch.tensor(self.adv[b], device=self.device),
                torch.tensor(self.ret[b], device=self.device),
            )
