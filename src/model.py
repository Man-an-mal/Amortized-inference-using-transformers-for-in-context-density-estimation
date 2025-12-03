"""Model definitions: simple transformer encoder + MDN heads."""
import torch
import torch.nn as nn

class SetTransformerMDN(nn.Module):
    def __init__(self, d_input=2+1+4, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, K_mdn=5, D=2, dropout=0.1):
        super().__init__()
        self.D = D
        self.K_mdn = K_mdn
        self.input_proj = nn.Linear(d_input, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pi_head = nn.Linear(d_model, K_mdn)
        self.mu_head = nn.Linear(d_model, K_mdn * D)
        self.logsig_head = nn.Linear(d_model, K_mdn * D)

    def forward(self, x):
        h = self.input_proj(x)
        h_enc = self.transformer(h)
        B, T, _ = h_enc.shape
        pi_logits = self.pi_head(h_enc)
        mu = self.mu_head(h_enc).view(B, T, self.K_mdn, self.D)
        logsig = self.logsig_head(h_enc).view(B, T, self.K_mdn, self.D)
        sig = torch.exp(logsig).clamp_min(1e-4)
        return pi_logits, mu, sig
