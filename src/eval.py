"""Evaluation utilities for the refactored ICDE codebase."""
import torch
from torch.utils.data import DataLoader
from src.datasets import EpisodicGMM, collate_with_gmmparams
from src.model import SetTransformerMDN
from src.utils import sample_from_mixture, episode_wasserstein_2d
import numpy as np

def load_model(checkpoint_path, device, args):
    d_input = args.D + 1 + 2*args.D
    model = SetTransformerMDN(d_input=d_input, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward, K_mdn=args.K_mdn, D=args.D).to(device)
    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck['model_state'])
    model.eval()
    return model

def evaluate(args):
    ds = EpisodicGMM(n_episodes=args.n_episodes, Nc=args.Nc, Nq=args.Nq, D=args.D, K_max=args.Kmax, normalize_episode=True, seed=args.seed)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_with_gmmparams)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device, args)
    wds = []
    for i, batch in enumerate(loader):
        Xc = batch['Xc']; Xq = batch['Xq']; ep = batch['ep_stats']; params = batch['gmm_params'][0]
        Xc = Xc.to(device); Xq = Xq.to(device); ep = ep.to(device)
        B = Xc.shape[0]
        m_ctx = torch.ones((B, Xc.shape[1], 1), device=Xc.device)
        ep_expand_ctx = ep.unsqueeze(1).expand(B, Xc.shape[1], ep.shape[-1])
        ctx_tokens = torch.cat([Xc, m_ctx, ep_expand_ctx], dim=-1)
        m_q = torch.zeros((B, Xq.shape[1], 1), device=Xc.device)
        ep_expand_q = ep.unsqueeze(1).expand(B, Xq.shape[1], ep.shape[-1])
        q_tokens = torch.cat([torch.zeros((B, Xq.shape[1], Xq.shape[2]), device=Xc.device), m_q, ep_expand_q], dim=-1)
        tokens = torch.cat([ctx_tokens, q_tokens], dim=1)
        with torch.no_grad():
            pi_logits, mu, sig = model(tokens)
        T = tokens.shape[1]; Nq = Xq.shape[1]
        pi_q = torch.softmax(pi_logits[:, T-Nq:, :], dim=-1).cpu().numpy().squeeze(0)
        mu_q = mu[:, T-Nq:, :, :].cpu().numpy().squeeze(0)
        sig_q = sig[:, T-Nq:, :, :].cpu().numpy().squeeze(0)
        avg_pis = pi_q.mean(axis=0)
        avg_means = mu_q.mean(axis=0)
        avg_stds = sig_q.mean(axis=0)
        pred = dict(pis=avg_pis, means=avg_means, stds=avg_stds)
        wd = episode_wasserstein_2d(params, pred, n_samples=2000)
        wds.append(wd)
    print('Average Wasserstein (approx):', np.mean(wds))
