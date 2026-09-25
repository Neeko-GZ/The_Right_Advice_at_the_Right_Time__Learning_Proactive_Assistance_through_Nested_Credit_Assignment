"""
advantages.py — Generalized Advantage Estimation.

Pure numerical code, no torch dependency on the rollout side: caller passes
in arrays of equal length T (one episode) and gets back arrays of equal
length T (advantages and returns).

GAE (Schulman et al. 2016, https://arxiv.org/abs/1506.02438):

    δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
    A_t = δ_t + γ · λ · (1 - done_t) · A_{t+1}
    R_t = A_t + V(s_t)         (the value-fn target for PPO)

Defaults: gamma=0.99, lam=0.95 — the standard PPO settings; safe to override
from train_config.

`bootstrap_value` is V(s_T), used for the last step's δ_T when the episode
is truncated (max_steps reached) rather than terminated (agent dead).
For terminated episodes, pass bootstrap_value=0 and dones[-1]=True.

Returns are typically standardized in PPO (mean=0, std=1) before the
advantage is fed to the policy loss; that's done in losses.py / train.py
not here. Here we keep the raw signal so logging / debugging stays clean.
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
    bootstrap_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE advantages and value-fn targets for a single trajectory.

    Args:
        rewards:  shape (T,)  per-step rewards (already shaped/normalized)
        values:   shape (T,)  V(s_t) under the current critic
        dones:    shape (T,)  bool — True iff the env was DONE after step t
                              (terminal, NOT truncation).
        gamma:    discount
        lam:      GAE lambda
        bootstrap_value: V(s_T) for the step *after* the last reward — used
                         when the trajectory was truncated, not terminated.
                         Pass 0 for a terminal episode.

    Returns:
        advantages: shape (T,)
        returns:    shape (T,)   (= advantages + values; the value-loss target)
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError(
            f"shape mismatch: rewards{rewards.shape} values{values.shape} "
            f"dones{dones.shape}"
        )

    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else bootstrap_value
        # If done at step t, the next-state value is masked out.
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def standardize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Mean-zero, unit-std. PPO usually applies this to advantages within a
    rollout (NOT across rollouts). Returns dtype float32."""
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + eps)
