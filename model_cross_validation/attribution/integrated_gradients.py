"""
attribution/integrated_gradients.py
======================================
Feature-attribution for the 6 sequence models: channel-wise Integrated
Gradients over the (3, T) input, using Captum where available, with a
manual 32-step Riemann-sum IG implementation as a fallback.

[PATCH 4 -- see docs/dev-log/step_ai_model_cross_validation.md] Calling
model.eval() (as done for out-of-fold prediction) and then calling
backward() through a cuDNN-backed nn.LSTM/nn.GRU (BiLSTM/BiGRU) for
Integrated Gradients raises
"RuntimeError: cudnn RNN backward can only be called in training mode",
because cuDNN's fused RNN kernel does not retain the intermediate tensors
backward needs while in eval mode. This is a kernel-selection limitation,
not a numerical issue, and does not affect CNN1D/TCN/TinyTransformer/
LightMamba (none use nn.LSTM/nn.GRU internally).

Fix: `compute_saliency()` wraps the attribution call (Captum or manual IG,
whichever path is used) in `torch.backends.cudnn.flags(enabled=False)`, so
only that call falls back to the generic (non-cuDNN) RNN backward
implementation. This does not change the model, switch it to training mode,
or affect numerics -- it only forces a slower, non-fused kernel for the
duration of the backward pass needed to compute gradients.
"""

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except (ImportError, OSError):
    HAVE_TORCH = False

HAVE_CAPTUM = False
if HAVE_TORCH:
    try:
        from captum.attr import IntegratedGradients
        HAVE_CAPTUM = True
    except (ImportError, OSError):
        HAVE_CAPTUM = False


def integrated_gradients_saliency(model, x, baseline=None, steps: int = 32):
    """Manual Integrated Gradients (used when captum is unavailable).
    baseline defaults to an all-zero tensor, which is the natural reference
    point here since each input channel is z-normalized per-window (see
    representations/raw_sequence_features.znorm) -- zero corresponds to
    "at the window's own mean", not an arbitrary absolute value.

    Must be called from within a `torch.backends.cudnn.flags(enabled=False)`
    context when the model contains cuDNN-backed RNN layers in eval mode
    (see module docstring / compute_saliency below) -- this function itself
    does not manage that context, since the flag is a process-global
    backend setting, not something local to a single call.
    """
    if not HAVE_TORCH:
        raise ImportError("torch is not installed; cannot compute Integrated Gradients.")
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)
    total_grad = torch.zeros_like(x)
    for alpha in np.linspace(0, 1, steps):
        xi = (baseline + alpha * (x - baseline)).clone().requires_grad_(True)
        out = model(xi).sum()
        grad, = torch.autograd.grad(out, xi, retain_graph=False)
        total_grad += grad
    avg_grad = total_grad / steps
    return (avg_grad * (x - baseline)).detach()


def compute_saliency(model, x, steps: int = 32) -> np.ndarray:
    """Top-level entry point used by run_all.py's sequence-model training
    loop. Returns a (3,) normalized channel-importance-share vector
    (level, diff, diff2), summed over time and averaged over the batch.

    [PATCH 4] The attribution call (Captum or manual fallback) is wrapped in
    `torch.backends.cudnn.flags(enabled=False)` so BiLSTM/BiGRU in eval mode
    do not crash on backward(). This applies uniformly to both the Captum
    and manual-IG code paths.
    """
    if not HAVE_TORCH:
        raise ImportError("torch is not installed; cannot compute Integrated Gradients.")

    with torch.backends.cudnn.flags(enabled=False):
        if HAVE_CAPTUM:
            ig = IntegratedGradients(model)
            attributions = ig.attribute(x, target=None, n_steps=steps)
        else:
            attributions = integrated_gradients_saliency(model, x, steps=steps)

    chan_importance = attributions.abs().sum(dim=2).mean(dim=0).detach().cpu().numpy()  # (3,)
    total = chan_importance.sum()
    return (chan_importance / total) if total > 0 else np.array([1 / 3, 1 / 3, 1 / 3])
