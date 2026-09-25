"""
value_head.py — independent MLP critic for PPO.

Why independent (not TRL-style ValueHead bolted onto Qwen):
  * The state we condition the critic on is the same 17-dim score feature
    the adapter consumes, NOT Qwen's hidden state. Qwen sees the soft tokens
    derived from the feature, but the critic doesn't need that detour — and
    keeping it small + on the raw feature avoids two failure modes:

    1. Coupling the value loss into the LoRA gradient path. With TRL's
       ValueHead, the value loss flows back through Qwen's last hidden
       state and its LoRA adapters, so a noisy V loss can perturb the
       policy. Separating them isolates that.

    2. Compute. Running Qwen in train() mode for the value pass alone is
       expensive; with an independent critic the value forward is ~µs
       and we only run Qwen for the policy logp pass.

  * Trivial to ablate: swap this for a TRL ValueHead in train.py if we
    later want to test whether Qwen's representation gives a better critic.

Architecture: 17 → 256 → SiLU → 256 → SiLU → 1. ~70K params, fp32.
That's 3 orders of magnitude smaller than the LoRA — the critic doesn't
dominate the optimizer.

The critic returns a scalar for each (B, 17) input: V(state).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Tiny MLP critic on the 17-dim score feature."""

    def __init__(
        self,
        input_dim: int = 17,
        hidden: int = 256,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth >= 1")
        layers: list[nn.Module] = []
        prev = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # Smaller init on the head — value targets are ~ episode return
        # magnitudes, typically in [-2, +5] for our reward design, so a
        # near-zero init prevents huge initial Adv spikes in PPO.
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) → (B,) scalar value."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        v = self.net(x).squeeze(-1)
        return v


def build_value_head(
    input_dim: int = 17,
    hidden: int = 256,
    depth: int = 2,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float32,
) -> ValueHead:
    head = ValueHead(input_dim=input_dim, hidden=hidden, depth=depth)
    head = head.to(device=device, dtype=dtype)
    return head
