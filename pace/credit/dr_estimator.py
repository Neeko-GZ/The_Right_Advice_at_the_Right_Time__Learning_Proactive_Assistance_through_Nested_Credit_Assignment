"""
dr_estimator.py — Doubly-Robust value estimation for the decision level.

See method.md § 5.3:

    V̂_d(s) = μ̂_d(s) + 1{a=d}/π_b(d|s) · (G − μ̂_d(s))

where
    - d ∈ {"HELP", "SILENCE"} is the decision we want to evaluate
    - μ̂_d(s) is the direct method (DM) estimate — e.g. critic prediction
      or branch-truth mean return at state s
    - π_b(d|s) is the behavior policy propensity, known exactly because
      we run ε-mixed policy ourselves: π_b(HELP|s) = (1−ε)p_help + ε·0.5
    - G is the observed MC return from state s
    - 1{a=d} is 1 iff the recorded action matched decision d

DR is unbiased if EITHER μ̂ is unbiased OR the IPS weight is correct
(here IPS is exact, so DR is always unbiased). Variance is minimized when
μ̂ is close to true V_d — the correction term (G - μ̂) shrinks toward 0.

This module provides three estimators over a batch of Transitions:
    - dm_estimate   : direct method only (uses μ̂ ignoring observations)
    - ips_estimate  : pure importance-sampling (uses G ignoring μ̂)
    - dr_estimate   : the recommended combination (μ̂ + IPS-corrected residual)

Plus per-state variants for policy gradient advantage computation, and a
convenience `dr_delta` for Δ = V_HELP − V_SILENCE.

Contract on Transition (see propensity_log.py for the full 20-field schema
— we use a subset):
    - state_hash        str
    - decision          "HELP" | "SILENCE"          (a — the taken action)
    - p_help_behavior   float in (0,1)              (π_b(HELP|s))
    - return_G          float                        (MC return from here on)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """Minimal transition for DR. Subset of propensity_log's Transition.

    Note `decision` is the RECORDED action (what actually happened), not the
    decision we're evaluating. The DR estimator internally checks
    1{decision == d_target} for each transition.
    """
    state_hash: str
    decision: str                # "HELP" | "SILENCE" — what happened
    p_help_behavior: float       # π_b(HELP|s) known exactly (ε-mixed policy)
    return_G: float              # MC return from this step onward
    # Optional: for logging / debugging, not required by estimators
    task_name: str = ""
    step: int = 0


# ---------------------------------------------------------------------------
# Propensity helpers
# ---------------------------------------------------------------------------

def epsilon_mix(p_help_policy: float, epsilon: float) -> float:
    """Behavior policy propensity: π_b(HELP|s) = (1−ε)·p_help + ε·0.5."""
    return (1.0 - epsilon) * p_help_policy + epsilon * 0.5


def p_b_of_decision(decision: str, p_help_behavior: float) -> float:
    """π_b(d|s) — behavior probability of the given decision at this state."""
    if decision == "HELP":
        return p_help_behavior
    if decision == "SILENCE":
        return 1.0 - p_help_behavior
    raise ValueError(f"unknown decision: {decision!r}")


# ---------------------------------------------------------------------------
# Per-state estimators
# ---------------------------------------------------------------------------

def dm_per_state(
    trans: Transition,
    mu_hat_fn: Callable[[str], float],
    d_target: str,
) -> float:
    """Direct method: V̂_d(s) = μ̂_d(s). Ignores the observed trajectory."""
    return float(mu_hat_fn(trans.state_hash))


def ips_per_state(
    trans: Transition,
    d_target: str,
    clip_max: Optional[float] = None,
) -> float:
    """Pure IPS: V̂_d(s) = 1{a=d}/π_b(d|s) · G. Zero when the recorded action
    didn't match d_target. Optional clip_max caps the IPS weight to bound
    variance at cost of bias."""
    if trans.decision != d_target:
        return 0.0
    pb = p_b_of_decision(d_target, trans.p_help_behavior)
    if pb <= 0:
        return 0.0
    w = 1.0 / pb
    if clip_max is not None:
        w = min(w, clip_max)
    return w * trans.return_G


def dr_per_state(
    trans: Transition,
    mu_hat_fn: Callable[[str], float],
    d_target: str,
    clip_max: Optional[float] = None,
) -> float:
    """DR: V̂_d(s) = μ̂_d(s) + 1{a=d}/π_b(d|s) · (G − μ̂_d(s))."""
    mu = float(mu_hat_fn(trans.state_hash))
    if trans.decision != d_target:
        return mu
    pb = p_b_of_decision(d_target, trans.p_help_behavior)
    if pb <= 0:
        return mu
    w = 1.0 / pb
    if clip_max is not None:
        w = min(w, clip_max)
    return mu + w * (trans.return_G - mu)


# ---------------------------------------------------------------------------
# Batch estimators — return scalar aggregate V_d over dataset
# ---------------------------------------------------------------------------

def dm_estimate(
    trans_list: Iterable[Transition],
    mu_hat_fn: Callable[[str], float],
    d_target: str,
) -> float:
    """E[μ̂_d(s)] over the dataset."""
    vals = [dm_per_state(t, mu_hat_fn, d_target) for t in trans_list]
    return sum(vals) / len(vals) if vals else 0.0


def ips_estimate(
    trans_list: Iterable[Transition],
    d_target: str,
    clip_max: Optional[float] = None,
) -> float:
    """E[1{a=d}/π_b(d|s) · G] over the dataset."""
    vals = [ips_per_state(t, d_target, clip_max=clip_max) for t in trans_list]
    return sum(vals) / len(vals) if vals else 0.0


def dr_estimate(
    trans_list: Iterable[Transition],
    mu_hat_fn: Callable[[str], float],
    d_target: str,
    clip_max: Optional[float] = None,
) -> float:
    """E[μ̂_d(s) + 1{a=d}/π_b(d|s) · (G − μ̂_d(s))] over the dataset."""
    vals = [dr_per_state(t, mu_hat_fn, d_target, clip_max=clip_max)
            for t in trans_list]
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Convenience: Δ = V_HELP − V_SILENCE
# ---------------------------------------------------------------------------

def dr_delta(
    trans_list: Iterable[Transition],
    mu_hat_fn: Callable[[str, str], float],
    clip_max: Optional[float] = None,
) -> dict:
    """Δ = V̂_HELP − V̂_SILENCE via DR on both arms.

    mu_hat_fn signature: (state_hash, decision) → float.
    Returns dict with keys V_HELP / V_SILENCE / delta / (and DM/IPS as diagnostics).
    """
    trans_list = list(trans_list)
    mu_help = lambda s: mu_hat_fn(s, "HELP")
    mu_silence = lambda s: mu_hat_fn(s, "SILENCE")

    v_help_dr = dr_estimate(trans_list, mu_help, "HELP", clip_max=clip_max)
    v_silence_dr = dr_estimate(trans_list, mu_silence, "SILENCE", clip_max=clip_max)

    # Diagnostics
    v_help_dm = dm_estimate(trans_list, mu_help, "HELP")
    v_silence_dm = dm_estimate(trans_list, mu_silence, "SILENCE")
    v_help_ips = ips_estimate(trans_list, "HELP", clip_max=clip_max)
    v_silence_ips = ips_estimate(trans_list, "SILENCE", clip_max=clip_max)

    return {
        "V_HELP_dr":     v_help_dr,
        "V_SILENCE_dr":  v_silence_dr,
        "delta_dr":      v_help_dr - v_silence_dr,
        "V_HELP_dm":     v_help_dm,
        "V_SILENCE_dm": v_silence_dm,
        "delta_dm":      v_help_dm - v_silence_dm,
        "V_HELP_ips":    v_help_ips,
        "V_SILENCE_ips": v_silence_ips,
        "delta_ips":     v_help_ips - v_silence_ips,
        "n_transitions": len(trans_list),
        "n_help":        sum(1 for t in trans_list if t.decision == "HELP"),
        "n_silence":     sum(1 for t in trans_list if t.decision == "SILENCE"),
    }


# ---------------------------------------------------------------------------
# Branch-truth backed μ̂ (for E3 ruler / PC2)
# ---------------------------------------------------------------------------

def make_branch_truth_mu_hat(
    branch_truth: dict,   # (state_hash, decision) → mean_return from K_seed branches
    fallback: float = 0.0,
) -> Callable:
    """Build a μ̂_d(s) lookup from brancher.branch_at_node results.

    Usage:
        # After running brancher on K nodes:
        truth = {}
        for node in nodes:
            results = brancher.branch_at_node(...)
            for br in results:
                mean = sum(br.seed_returns) / len(br.seed_returns)
                truth[(node.state_hash, br.decision)] = mean

        mu_hat = make_branch_truth_mu_hat(truth)
        v_help = dr_estimate(rollout_trans, mu_hat, "HELP")

    States not in branch_truth get `fallback` (0.0 by default — biases toward
    zero, which under DR gets corrected by IPS term anyway).
    """
    def fn(state_hash, decision=None):
        if decision is None:
            # Called from dm/ips/dr per_state without decision (legacy);
            # can't route — return fallback
            return fallback
        return branch_truth.get((state_hash, decision), fallback)
    return fn
