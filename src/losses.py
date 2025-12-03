"""Loss functions for MDN and helpers."""
import torch
import torch.nn.functional as F

EPS = 1e-8

def mdn_nll_gaussian(pi_logits, mu, sig, x_target):
    B, Tq, K = pi_logits.shape
    D = x_target.shape[-1]
    x = x_target.unsqueeze(2).expand(B, Tq, K, D)
    log_component = -0.5 * (((x - mu) / (sig + EPS))**2).sum(dim=-1) - (D/2.) * torch.log(2 * torch.pi) - torch.log(sig + EPS).sum(dim=-1)
    log_weights = F.log_softmax(pi_logits, dim=-1)
    log_mix = torch.logsumexp(log_weights + log_component, dim=-1)
    nll = - log_mix.mean()
    return nll
