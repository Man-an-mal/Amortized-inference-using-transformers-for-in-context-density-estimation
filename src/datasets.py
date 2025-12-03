"""Datasets for ICDE: Episodic GMM generator and collate functions."""
from typing import Tuple, Dict, Any
import numpy as np
from torch.utils.data import Dataset
import torch

class EpisodicGMM(Dataset):
    """Generate episodes where each episode is a 2-D Gaussian mixture model."""
    def __init__(self, n_episodes=1000, Nc=32, Nq=64, D=2, K_max=5, normalize_episode=True, seed=0):
        self.n_episodes = n_episodes
        self.Nc = Nc
        self.Nq = Nq
        self.D = D
        self.K_max = K_max
        self.normalize_episode = normalize_episode
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n_episodes

    def _sample_gmm(self):
        K = int(self.rng.randint(2, self.K_max+1))
        pis = self.rng.dirichlet(alpha=np.ones(K))
        means = self.rng.normal(loc=0.0, scale=5.0, size=(K, self.D))
        stds = self.rng.uniform(0.3, 2.0, size=(K, self.D))
        return dict(K=K, pis=pis, means=means, stds=stds)

    def _sample_from_gmm(self, params, n):
        K = params['K']
        pis = params['pis']
        means = params['means']
        stds = params['stds']
        comps = self.rng.choice(K, size=n, p=pis)
        samples = np.zeros((n, self.D), dtype=np.float32)
        for i, c in enumerate(comps):
            samples[i] = self.rng.normal(loc=means[c], scale=stds[c])
        return samples

    def __getitem__(self, idx):
        params = self._sample_gmm()
        Xc = self._sample_from_gmm(params, self.Nc)
        Xq = self._sample_from_gmm(params, self.Nq)
        mu_ctx = Xc.mean(axis=0, keepdims=True)
        std_ctx = Xc.std(axis=0, keepdims=True) + 1e-6
        ep_stats = np.concatenate([mu_ctx, std_ctx], axis=-1).squeeze(0).astype(np.float32)
        if self.normalize_episode:
            Xc_n = ((Xc - mu_ctx) / std_ctx).astype(np.float32)
            Xq_n = ((Xq - mu_ctx) / std_ctx).astype(np.float32)
            return Xc_n, Xq_n, ep_stats, params
        else:
            return Xc.astype(np.float32), Xq.astype(np.float32), ep_stats, params

def collate_for_train(batch):
    Xc = np.stack([b[0] for b in batch], axis=0)
    Xq = np.stack([b[1] for b in batch], axis=0)
    ep_stats = np.stack([b[2] for b in batch], axis=0)
    Xc_t = torch.from_numpy(Xc)
    Xq_t = torch.from_numpy(Xq)
    ep_t = torch.from_numpy(ep_stats)
    return dict(Xc=Xc_t, Xq=Xq_t, ep_stats=ep_t)

def collate_with_gmmparams(batch):
    Xc = np.stack([b[0] for b in batch], axis=0)
    Xq = np.stack([b[1] for b in batch], axis=0)
    ep_stats = np.stack([b[2] for b in batch], axis=0)
    params = [b[3] for b in batch]
    Xc_t = torch.from_numpy(Xc)
    Xq_t = torch.from_numpy(Xq)
    ep_t = torch.from_numpy(ep_stats)
    return dict(Xc=Xc_t, Xq=Xq_t, ep_stats=ep_t, gmm_params=params)
