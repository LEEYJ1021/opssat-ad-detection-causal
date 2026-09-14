"""
models/attention_sequence.py
==============================
Attention family: TinyTransformer -- a small 2-layer, 4-head
TransformerEncoder with fixed sinusoidal positional encoding. Raw-sequence
representation, input (B, 3, T) -> output (B,) logit.
"""

import math

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except (ImportError, OSError):
    HAVE_TORCH = False


if HAVE_TORCH:

    class TinyTransformer(nn.Module):
        def __init__(self, c_in: int = 3, d_model: int = 32, nhead: int = 4,
                     num_layers: int = 2, max_len: int = 200):
            super().__init__()
            self.proj = nn.Linear(c_in, d_model)

            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            xt = x.permute(0, 2, 1)  # (B, T, 3)
            h = self.proj(xt) + self.pe[:, :xt.size(1), :]
            h = self.encoder(h)
            pooled = h.mean(dim=1)
            return self.head(pooled).squeeze(-1)

else:
    TinyTransformer = None


def build_models():
    if not HAVE_TORCH:
        raise ImportError(
            "torch is not installed; the Attention family cannot be built. "
            "Install with: pip install torch"
        )
    return [
        ("TinyTransformer", "Attention", TinyTransformer),
    ]
