"""
losses.py — PPO loss components.

We keep everything that touches torch/grads here; everything in advantages.py
is numpy. This split makes the math testable in isolation (advantages.py)
and the gradient path self-contained (losses.py).

Three losses:
  1. Clipped policy loss (Schulman et al. 2017)
        L_clip = E[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ]

  2. Clipped value loss (PPO2 / OpenAI baselines)
        v_clipped = v_old + clip(v_new - v_old, -ε_v, +ε_v)
        L_v = 0.5 * max( (v_new - R)^2, (v_clipped - R)^2 )

  3. KL penalty against frozen reference policy (InstructGPT-style)
        L_kl = β * KL(π_θ || π_ref)
     Approximated as the per-token logp difference, averaged over response
     tokens; this is the cheap k1 estimator (∼ logp_old - logp_ref).

Helper utilities:
  * `gather_response_logprobs(logits, response_ids, response_start)` — picks
    out the logp of each response token under the given logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class PPOLossConfig:
    clip_range: float = 0.2          # ε for policy ratio clip
    clip_range_vf: float = 0.2       # ε_v for value clip
    vf_coef: float = 0.5             # value loss weight in total
    ent_coef: float = 0.0            # entropy bonus (rarely needed for LM)
    kl_coef: float = 0.05            # InstructGPT-style KL penalty β
    target_kl: Optional[float] = 0.05  # early-stop epoch if KL exceeds this


# ---------------------------------------------------------------------------
# log-prob extraction
# ---------------------------------------------------------------------------

def gather_response_logprobs(
    logits: torch.Tensor,         # (B, T, V)
    response_ids: torch.Tensor,   # (B, R)
    response_start: int,          # logit index s.t. logits[:, rs - 1] predicts response_ids[:, 0]
) -> torch.Tensor:
    """
    Returns per-token logp of the response under the given logits, shape (B, R).

    Note the off-by-one: position (rs - 1) predicts the FIRST response token,
    so we slice logits[:, rs-1 : rs-1+R, :] and gather along V at response_ids.
    """
    B, T, V = logits.shape
    Rt = response_ids.shape[1]
    if response_start - 1 + Rt > T:
        raise ValueError(
            f"logits seq {T} can't cover response_start={response_start} + R={Rt}"
        )
    pred_logits = logits[:, response_start - 1 : response_start - 1 + Rt, :]   # (B, R, V)
    logp = F.log_softmax(pred_logits.float(), dim=-1)                          # (B, R, V)
    # Gather logp at the actual response token ids
    token_logp = logp.gather(-1, response_ids.unsqueeze(-1).to(logp.device)).squeeze(-1)
    return token_logp                                                          # (B, R)


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean over a possibly-masked tensor. mask: same shape as x, 1 = keep, 0 = ignore."""
    if mask is None:
        return x.mean()
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def policy_loss(
    logp_new: torch.Tensor,    # (B, R) per-token logp under current policy
    logp_old: torch.Tensor,    # (B, R) per-token logp at rollout time (no grad)
    advantages: torch.Tensor,  # (B,)   per-sequence advantage
    cfg: PPOLossConfig,
    response_mask: Optional[torch.Tensor] = None,  # (B, R) mask
) -> tuple[torch.Tensor, dict]:
    """
    Sequence-level PPO clip on the response. We aggregate per-token logp into
    a per-sequence sum-logp before computing the ratio — this matches the
    standard "treat the full response as one action" setup used in RLHF.
    """
    if response_mask is not None:
        seq_logp_new = (logp_new * response_mask).sum(dim=-1)
        seq_logp_old = (logp_old * response_mask).sum(dim=-1)
    else:
        seq_logp_new = logp_new.sum(dim=-1)
        seq_logp_old = logp_old.sum(dim=-1)

    log_ratio = seq_logp_new - seq_logp_old
    ratio = log_ratio.exp()                                  # (B,)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * advantages
    loss = -torch.min(surr1, surr2).mean()

    with torch.no_grad():
        clipfrac = ((ratio - 1.0).abs() > cfg.clip_range).float().mean()
        approx_kl = (-log_ratio).mean()                      # k1 estimator

    return loss, {
        "loss/policy": float(loss.item()),
        "ppo/ratio_mean": float(ratio.mean().item()),
        "ppo/ratio_min": float(ratio.min().item()),
        "ppo/ratio_max": float(ratio.max().item()),
        "ppo/clipfrac": float(clipfrac.item()),
        "ppo/approx_kl_old_new": float(approx_kl.item()),
    }


def value_loss(
    v_new: torch.Tensor,    # (B,)
    v_old: torch.Tensor,    # (B,) value at rollout time, no grad
    returns: torch.Tensor,  # (B,) target = adv + v_old (returns)
    cfg: PPOLossConfig,
) -> tuple[torch.Tensor, dict]:
    """Clipped value loss (PPO2-style)."""
    v_clipped = v_old + torch.clamp(v_new - v_old, -cfg.clip_range_vf, cfg.clip_range_vf)
    loss_unclipped = (v_new - returns).pow(2)
    loss_clipped = (v_clipped - returns).pow(2)
    loss = 0.5 * torch.max(loss_unclipped, loss_clipped).mean()
    with torch.no_grad():
        clipfrac = ((v_new - v_old).abs() > cfg.clip_range_vf).float().mean()
    return loss, {
        "loss/value": float(loss.item()),
        "value/clipfrac": float(clipfrac.item()),
        "value/mean": float(v_new.mean().item()),
        "value/target_mean": float(returns.mean().item()),
    }


def kl_against_ref(
    logp_new: torch.Tensor,    # (B, R)
    logp_ref: torch.Tensor,    # (B, R), no grad — frozen ref policy
    response_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Per-token KL k1 estimator: logp_new - logp_ref, masked-mean over response.
    Negative => ref assigns higher prob; positive => current diverged from ref.
    Loss = mean (we want this small, hence add to total with kl_coef * mean).
    """
    diff = logp_new - logp_ref
    kl = masked_mean(diff, response_mask)
    return kl, {
        "ppo/kl_to_ref": float(kl.detach().item()),
    }


# ---------------------------------------------------------------------------
# total
# ---------------------------------------------------------------------------

def ppo_total_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    advantages: torch.Tensor,
    v_new: torch.Tensor,
    v_old: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOLossConfig,
    response_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Sum the three terms with their coefficients. Returns (loss, log_dict)."""
    pol_loss, pol_log = policy_loss(logp_new, logp_old, advantages, cfg, response_mask)
    val_loss, val_log = value_loss(v_new, v_old, returns, cfg)
    kl, kl_log = kl_against_ref(logp_new, logp_ref, response_mask)

    total = pol_loss + cfg.vf_coef * val_loss + cfg.kl_coef * kl

    log = {
        **pol_log, **val_log, **kl_log,
        "loss/kl_weighted": float((cfg.kl_coef * kl).item()),
        "loss/total": float(total.item()),
    }
    return total, log
