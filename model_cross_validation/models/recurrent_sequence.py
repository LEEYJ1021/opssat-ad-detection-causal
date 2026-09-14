"""
models/recurrent_sequence.py
==============================
Recurrent family: BiLSTM and BiGRU. Raw-sequence representation, input
(B, 3, T) -> permuted to (B, T, 3) -> output (B,) logit.

NOTE on cuDNN backward-mode: when these models are placed in eval() mode
(as they are for out-of-fold prediction) and then subjected to a
backward() call for Integrated-Gradients attribution on a cuDNN/GPU
backend, cuDNN's fused RNN kernel raises
"RuntimeError: cudnn RNN backward can only be called in training mode"
because it does not retain the intermediate tensors backward needs in eval
mode. This is a kernel-selection issue, not a numerical one. The fix
(wrapping the attribution call in `torch.backends.cudnn.flags(enabled=False)`)
lives in attribution/integrated_gradients.py, not here, since it only needs
to apply for the duration of the attribution call -- see that module's
docstring for the full explanation. CNN1D/TCN/TinyTransformer/LightMamba do
not use nn.LSTM/nn.GRU internally and are unaffected.
"""

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except (ImportError, OSError):
    HAVE_TORCH = False


if HAVE_TORCH:

    class BiLSTM(nn.Module):
        def __init__(self, c_in: int = 3, hidden: int = 32):
            super().__init__()
            self.rnn = nn.LSTM(c_in, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Linear(hidden * 2, 1)

        def forward(self, x):
            xt = x.permute(0, 2, 1)  # (B, T, 3)
            out, _ = self.rnn(xt)
            pooled = out.mean(dim=1)
            return self.head(pooled).squeeze(-1)

    class BiGRU(nn.Module):
        def __init__(self, c_in: int = 3, hidden: int = 32):
            super().__init__()
            self.rnn = nn.GRU(c_in, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Linear(hidden * 2, 1)

        def forward(self, x):
            xt = x.permute(0, 2, 1)
            out, _ = self.rnn(xt)
            pooled = out.mean(dim=1)
            return self.head(pooled).squeeze(-1)

else:
    BiLSTM = None
    BiGRU = None


def build_models():
    if not HAVE_TORCH:
        raise ImportError(
            "torch is not installed; the Recurrent family cannot be built. "
            "Install with: pip install torch"
        )
    return [
        ("BiLSTM", "Recurrent", BiLSTM),
        ("BiGRU", "Recurrent", BiGRU),
    ]
