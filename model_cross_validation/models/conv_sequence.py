"""
models/conv_sequence.py
=========================
Convolutional / Convolutional (dilated causal) families: CNN1D and TCN.
Raw-sequence representation, input (B, 3, T) -> output (B,) logit.
Attribution: Integrated Gradients over the 3 input channels
(attribution/integrated_gradients.py).
"""

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except (ImportError, OSError):
    HAVE_TORCH = False


if HAVE_TORCH:

    class CNN1D(nn.Module):
        """Two plain 1D convolutions + global average pooling + linear head."""

        def __init__(self, c_in: int = 3, hidden: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(c_in, hidden, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.net(x).squeeze(-1)
            return self.head(h).squeeze(-1)

    class TCNBlock(nn.Module):
        """Dilated causal convolution residual block."""

        def __init__(self, c_in: int, c_out: int, dilation: int):
            super().__init__()
            pad = (5 - 1) * dilation
            self.conv = nn.Conv1d(c_in, c_out, kernel_size=5, padding=pad, dilation=dilation)
            self.pad = pad
            self.relu = nn.ReLU()
            self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        def forward(self, x):
            out = self.conv(x)[:, :, :-self.pad] if self.pad > 0 else self.conv(x)
            return self.relu(out + self.res(x))

    class TCN(nn.Module):
        """3-block dilated causal TCN (dilations 1, 2, 4)."""

        def __init__(self, c_in: int = 3, hidden: int = 32):
            super().__init__()
            self.blocks = nn.Sequential(
                TCNBlock(c_in, hidden, dilation=1),
                TCNBlock(hidden, hidden, dilation=2),
                TCNBlock(hidden, hidden, dilation=4),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.blocks(x)
            h = self.pool(h).squeeze(-1)
            return self.head(h).squeeze(-1)

else:
    CNN1D = None
    TCN = None


def build_models():
    """Return [(name, family, ModelClass), ...] -- ModelClass, not an
    instance, since sequence models are re-instantiated fresh per fold."""
    if not HAVE_TORCH:
        raise ImportError(
            "torch is not installed; the Convolutional/Convolutional "
            "(dilated causal) families cannot be built. Install with: "
            "pip install torch"
        )
    return [
        ("CNN1D", "Convolutional", CNN1D),
        ("TCN", "Convolutional(dilated causal)", TCN),
    ]
