"""Training script for ICDE. Run with: python -m src.train"""
import argparse
import torch
from torch.utils.data import DataLoader
from src.datasets import EpisodicGMM, collate_for_train
from src.model import SetTransformerMDN
from src.losses import mdn_nll_gaussian
import torch.nn.utils as nn_utils

def train(args):
    ds = EpisodicGMM(n_episodes=args.n_episodes, Nc=args.Nc, Nq=args.Nq, D=args.D, K_max=args.Kmax, normalize_episode=True, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_for_train, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_input = args.D + 1 + 2*args.D
    model = SetTransformerMDN(d_input=d_input, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward, K_mdn=args.K_mdn, D=args.D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            Xc = batch['Xc'].to(device)
            Xq = batch['Xq'].to(device)
            ep = batch['ep_stats'].to(device)
            B = Xc.shape[0]
            def build_tokens(Xc, Xq, ep):
                B, Nc, D = Xc.shape
                Nq = Xq.shape[1]
                m_ctx = torch.ones((B, Nc, 1), device=Xc.device)
                ep_expand_ctx = ep.unsqueeze(1).expand(B, Nc, ep.shape[-1])
                ctx_tokens = torch.cat([Xc, m_ctx, ep_expand_ctx], dim=-1)
                m_q = torch.zeros((B, Nq, 1), device=Xc.device)
                ep_expand_q = ep.unsqueeze(1).expand(B, Nq, ep.shape[-1])
                q_tokens = torch.cat([torch.zeros((B, Nq, D), device=Xc.device), m_q, ep_expand_q], dim=-1)
                tokens = torch.cat([ctx_tokens, q_tokens], dim=1)
                return tokens
            tokens = build_tokens(Xc, Xq, ep)
            pi_logits, mu, sig = model(tokens)
            T = tokens.shape[1]; Nq = Xq.shape[1]
            pi_q = pi_logits[:, T-Nq:, :]
            mu_q = mu[:, T-Nq:, :, :]
            sig_q = sig[:, T-Nq:, :, :]
            loss = mdn_nll_gaussian(pi_q, mu_q, sig_q, Xq)
            opt.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); n_batches += 1
        avg = total_loss / max(1, n_batches)
        print(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(dict(model_state=model.state_dict(), opt_state=opt.state_dict(), epoch=epoch), args.save_path)
    print("Training complete. Best loss:", best_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', type=int, default=2000)
    parser.add_argument('--Nc', type=int, default=32)
    parser.add_argument('--Nq', type=int, default=64)
    parser.add_argument('--D', type=int, default=2)
    parser.add_argument('--Kmax', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--d-model', dest='d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dim-feedforward', dest='dim_feedforward', type=int, default=256)
    parser.add_argument('--K-mdn', dest='K_mdn', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-path', type=str, default='ckpt_best.pt')
    args = parser.parse_args()
    train(args)
