"""
models/ssm_sequence.py
========================
State-space (SSM) family: LightMamba -- a pure-PyTorch selective
state-space block standing in for the official `mamba_ssm` CUDA-kernel
package, which is unavailable in this environment. Raw-sequence
representation, input (B, 3, T) -> output (B,) logit.

This block keeps only Mamba's core idea -- state-transition dynamics that
depend on the input at each timestep ("selective") -- implemented as an
explicit, non-fused sequential scan. It does not reproduce the official
implementation's speed or capacity, and this analysis uses it strictly as a
minimal working stand-in to check whether the SSM inductive bias shows a
feature-attribution pattern (level vs. diff/diff2) similar to the other 12
families. If mamba_ssm is installed, this class should be swapped for the
official Mamba block and the model-cross-validation run repeated to confirm
the result under the real implementation (see README limitations).

Result under this substitute (see agreement_statistics output): SSM shows
the narrowest margin of all 13 families (level share = 0.428 vs.
diff+diff2 = 0.571) -- flagged for confirmation, not treated as a
counter-example.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except (ImportError, OSError):
    HAVE_TORCH = False


if HAVE_TORCH:

    class LightweightMambaBlock(nn.Module):
        """Minimal selective state-space block (sequential scan, no fused
        CUDA kernel). See module docstring for scope and caveats."""

        def __init__(self, c_in: int = 3, d_model: int = 32, d_state: int = 16):
            super().__init__()
            self.proj_in = nn.Linear(c_in, d_model)
            self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.1)  # -exp(A_log) < 0 for stability
            self.B_proj = nn.Linear(d_model, d_state)
            self.C_proj = nn.Linear(d_model, d_state)
            self.delta_proj = nn.Linear(d_model, d_model)
            self.head = nn.Linear(d_model, 1)
            self.d_model, self.d_state = d_model, d_state

        def forward(self, x):
            xt = x.permute(0, 2, 1)  # (B, T, 3)
            u = self.proj_in(xt)  # (B, T, d_model)
            B_, T_, D = u.shape

            delta = F.softplus(self.delta_proj(u))       # (B, T, D) input-dependent timestep
            A = -torch.exp(self.A_log)                    # (D, d_state), always negative (stability)
            Bc = self.B_proj(u)                            # (B, T, d_state) input-dependent B
            Cc = self.C_proj(u)                            # (B, T, d_state) input-dependent C

            h = torch.zeros(B_, D, self.d_state, device=u.device)
            ys = []
            for t in range(T_):
                dA = torch.exp(delta[:, t, :].unsqueeze(-1) * A.unsqueeze(0))  # (B, D, d_state)
                dBu = delta[:, t, :].unsqueeze(-1) * Bc[:, t, :].unsqueeze(1) * u[:, t, :].unsqueeze(-1)
                h = dA * h + dBu  # selective state update
                y_t = (h * Cc[:, t, :].unsqueeze(1)).sum(-1)  # (B, D)
                ys.append(y_t)
            y = torch.stack(ys, dim=1)  # (B, T, D)
            pooled = y.mean(dim=1)
            return self.head(pooled).squeeze(-1)

else:
    LightweightMambaBlock = None


def build_models():
    if not HAVE_TORCH:
        raise ImportError(
            "torch is not installed; the State-space (SSM) family cannot "
            "be built. Install with: pip install torch"
        )
    return [
        ("LightMamba", "State-space(SSM)", LightweightMambaBlock),
    ]
