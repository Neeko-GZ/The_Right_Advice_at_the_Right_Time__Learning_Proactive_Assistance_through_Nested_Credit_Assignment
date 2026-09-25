"""
cf_advantage.py — K+1 group-relative advantage per HELP state.

Method.md § 4.2 defines the CF-nstep-GRPO advantage as:

    Â_i = (Q_i - Q̄) / σ_Q  +  α_intr · r_intr(a_i)  +  α_div · r_div(a_i, {a_j})

where the K+1 group at each HELP state consists of:
  * 1 SILENCE branch  (idx 0)      — advice="", forced gate=SILENCE
  * K HELP branches   (idx 1..K)  — fresh sampled advice
                                      + main-traj advice (HELP-replay branch)

The SILENCE branch is included as a full group member — its Q_i (via n-step
return under snapshot+CRN) is directly comparable to the HELP branches
because they all start from the same state and share seed. This IS the gate
counterfactual signal: if Q_silence >> Q̄_help, the scheduler learns SILENCE
was better; conversely Q_help > Q_silence teaches HELP.

Content-side training gets the K+1 advantages directly attributed to each
branch's advice tokens (via cf_token_masks).

Gate-side training uses a scalar single-branch (replay) signal per state:
    A_gate(s_t) = (Q_help_replay - Q_silence) / σ_Q
i.e. the counterfactual value of the advice the gate ACTUALLY produced (the
HELP-replay branch, idx 1) vs SILENCE — not an average over the K fresh advice
the gate did not pick. Matches the framework figure and method.md §4.2.

Pure numerical code, no torch dependency. Takes lists / arrays as input.

Reference: method.md § 4.2, § 4.3, § 5.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GroupAdvantage:
    """Result of computing K+1 group-relative advantages at one HELP state.

    Fields:
      q_values         length K+1, raw n-step Q per branch (index 0 = SILENCE)
      q_mean           group mean Q̄
      q_std            group std σ_Q (with std_eps floor)
      advantages       length K+1, final Â_i after shaping added

      # Diagnostics / breakdown (all length K+1):
      q_normalized     (Q_i - Q̄)/σ_Q  — the pure GRPO part
      r_intrinsic      per-branch intrinsic reward (advice quality)
      r_diversity      per-branch diversity reward

      # Gate-side signal:
      gate_advantage   scalar A_gate(s_t) for training the gate token

      # Config used (for logging):
      alpha_intr, alpha_div, std_eps
    """
    q_values: List[float]
    q_mean: float
    q_std: float
    advantages: List[float]
    q_normalized: List[float]
    r_intrinsic: List[float]
    r_diversity: List[float]
    gate_advantage: float
    alpha_intr: float
    alpha_div: float
    std_eps: float


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_group_advantages(
    q_values: Sequence[float],
    r_intrinsic: Sequence[float],
    r_diversity: Sequence[float],
    silence_index: int = 0,
    replay_index: int = 1,
    alpha_intr: float = 0.1,
    alpha_div: float = 0.05,
    std_eps: float = 1e-6,
    min_std: float = 0.15,
    gate_adv_clip: float = 2.0,
) -> GroupAdvantage:
    """Compute K+1 group-relative advantages + gate-side scalar.

    Args:
      q_values          length K+1, n-step Q per branch. Index `silence_index`
                        is the SILENCE branch; all others are HELP branches.
      r_intrinsic       length K+1, intrinsic quality reward per branch
      r_diversity       length K+1, diversity reward per branch
      silence_index     which index is the SILENCE branch (default 0)
      alpha_intr        weight α_intr for intrinsic shaping in advantage
      alpha_div         weight α_div for diversity shaping in advantage
      std_eps           numerical floor for σ_Q to avoid div-by-zero
      min_std           minimum σ_Q used for normalization. When the group
                        is nearly degenerate (all Q equal), we clamp σ_Q up
                        so that shaping terms don't get artificially amplified.
                        Set to 0.15 (W32): the original 0.1 floor let the gate
                        advantage inflate to ~10x with tight branch returns
                        (rare success => most Q~0), driving the KL blowup; a
                        brief 0.3 over-corrected and starved the gate signal.
                        0.15 + the gate_adv_clip cap balances the two.
      gate_adv_clip     hard magnitude cap on the gate-side scalar advantage.
                        Even after the min_std floor, a single-token gate loss
                        is very sensitive; clamp to +/-gate_adv_clip so one
                        HELP state cannot slam the gate logit.

    Returns:
      GroupAdvantage with all fields populated.
    """
    n = len(q_values)
    if n < 2:
        raise ValueError(f"need at least 2 branches for group-relative, got {n}")
    if len(r_intrinsic) != n or len(r_diversity) != n:
        raise ValueError(
            f"length mismatch: q_values={n}, r_intrinsic={len(r_intrinsic)}, "
            f"r_diversity={len(r_diversity)}"
        )
    if not (0 <= silence_index < n):
        raise ValueError(f"silence_index {silence_index} out of range [0,{n})")

    q_list = [float(x) for x in q_values]
    q_mean = sum(q_list) / n

    # Population std (matches GRPO practice)
    variance = sum((q - q_mean) ** 2 for q in q_list) / n
    q_std_raw = math.sqrt(variance)
    q_std = max(q_std_raw, min_std, std_eps)

    q_normalized = [(q - q_mean) / q_std for q in q_list]

    advantages = [
        q_normalized[i]
        + alpha_intr * float(r_intrinsic[i])
        + alpha_div * float(r_diversity[i])
        for i in range(n)
    ]

    # Gate-side signal (single-branch / replay):
    #   A_gate = (Q_help_replay - Q_silence) / σ_Q.
    #   The gate at s_t actually produced the *replayed* advice a_t, so its
    #   counterfactual value is that specific advice vs SILENCE — NOT an average
    #   over the K fresh advice the gate did not pick (which would inject advice-
    #   sampling noise into the gate credit). Matches the framework figure and
    #   method.md §4.2: do(HELP)=replay vs do(SILENCE).
    q_silence = q_list[silence_index]
    if 0 <= replay_index < n and replay_index != silence_index:
        q_help = q_list[replay_index]
    else:
        # Fallback (no distinct replay branch): mean over all HELP branches.
        help_indices = [i for i in range(n) if i != silence_index]
        q_help = sum(q_list[i] for i in help_indices) / len(help_indices)
    gate_advantage = (q_help - q_silence) / q_std
    # Hard magnitude cap (W32 fix): keep one HELP state from slamming the
    # single gate token even if the normalized signal is large.
    gate_advantage = max(-gate_adv_clip, min(gate_adv_clip, gate_advantage))

    return GroupAdvantage(
        q_values=q_list,
        q_mean=q_mean,
        q_std=q_std,
        advantages=advantages,
        q_normalized=q_normalized,
        r_intrinsic=[float(x) for x in r_intrinsic],
        r_diversity=[float(x) for x in r_diversity],
        gate_advantage=gate_advantage,
        alpha_intr=alpha_intr,
        alpha_div=alpha_div,
        std_eps=std_eps,
    )


# ---------------------------------------------------------------------------
# Batching helper
# ---------------------------------------------------------------------------

@dataclass
class HelpStateAdvantageBatch:
    """Collate a batch of GroupAdvantages, one per HELP state, into flat
    tensors-friendly arrays for the PPO trainer.

    Fields:
      per_state           list of GroupAdvantage
      # Flat content-side arrays (each length = num_states * (K+1))
      flat_advantages     for content PPO loss weighting
      flat_state_ids      which HELP state (in the list) each entry came from
      flat_branch_ids     which branch (0..K) within its state
      # Flat gate-side arrays (length = num_states)
      gate_advantages     one scalar per HELP state
    """
    per_state: List[GroupAdvantage]
    flat_advantages: List[float]
    flat_state_ids: List[int]
    flat_branch_ids: List[int]
    gate_advantages: List[float]

    @property
    def num_help_states(self) -> int:
        return len(self.per_state)

    @property
    def num_branch_samples(self) -> int:
        return len(self.flat_advantages)


def batch_group_advantages(
    per_state_inputs: Sequence[Tuple[Sequence[float], Sequence[float], Sequence[float]]],
    silence_index: int = 0,
    replay_index: int = 1,
    alpha_intr: float = 0.1,
    alpha_div: float = 0.05,
    std_eps: float = 1e-6,
    min_std: float = 0.15,
) -> HelpStateAdvantageBatch:
    """Compute group advantages for a batch of HELP states.

    Args:
      per_state_inputs: list of (q_values, r_intrinsic, r_diversity) tuples,
                        one per HELP state, each with K+1 branches.

    Returns:
      HelpStateAdvantageBatch with per-state GroupAdvantages + flat arrays.
    """
    per_state: List[GroupAdvantage] = []
    flat_adv: List[float] = []
    flat_sid: List[int] = []
    flat_bid: List[int] = []
    gate_adv: List[float] = []

    for sid, (qs, ris, rds) in enumerate(per_state_inputs):
        ga = compute_group_advantages(
            qs, ris, rds,
            silence_index=silence_index, replay_index=replay_index,
            alpha_intr=alpha_intr, alpha_div=alpha_div,
            std_eps=std_eps, min_std=min_std,
        )
        per_state.append(ga)
        for bid, a in enumerate(ga.advantages):
            flat_adv.append(a)
            flat_sid.append(sid)
            flat_bid.append(bid)
        gate_adv.append(ga.gate_advantage)

    return HelpStateAdvantageBatch(
        per_state=per_state,
        flat_advantages=flat_adv,
        flat_state_ids=flat_sid,
        flat_branch_ids=flat_bid,
        gate_advantages=gate_adv,
    )


# ---------------------------------------------------------------------------
# Sanity self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    print("[test] Case 1: HELP clearly better than SILENCE (K=3, +1 silence)")
    # SILENCE:0.2, HELP branches: 0.8, 0.9, 0.7 → HELP wins by ~0.6
    ga = compute_group_advantages(
        q_values=[0.2, 0.8, 0.9, 0.7],
        r_intrinsic=[0.4, 0.76, 1.0, 0.76],
        r_diversity=[1.0, 0.9, 0.9, 0.94],
        alpha_intr=0.1, alpha_div=0.05,
    )
    print(f"  Q values: {ga.q_values}")
    print(f"  Q̄={ga.q_mean:.3f}  σ_Q={ga.q_std:.3f}")
    print(f"  Q_normalized: {[f'{q:+.3f}' for q in ga.q_normalized]}")
    print(f"  Advantages:   {[f'{a:+.3f}' for a in ga.advantages]}")
    print(f"  gate_advantage: {ga.gate_advantage:+.3f}  (expect >>0)")
    # SILENCE index 0 should have the most negative q_normalized
    assert ga.q_normalized[0] == min(ga.q_normalized)
    # Gate advantage should strongly favor HELP
    assert ga.gate_advantage > 0.5

    print("[test] Case 2: SILENCE better than HELP")
    ga2 = compute_group_advantages(
        q_values=[0.9, 0.3, 0.2, 0.4],
        r_intrinsic=[0.4, 0.76, 1.0, 0.76],
        r_diversity=[1.0, 0.9, 0.9, 0.94],
    )
    print(f"  Advantages: {[f'{a:+.3f}' for a in ga2.advantages]}")
    print(f"  gate_advantage: {ga2.gate_advantage:+.3f}  (expect <0)")
    assert ga2.gate_advantage < -0.5
    # SILENCE should now have the most positive q_normalized
    assert ga2.q_normalized[0] == max(ga2.q_normalized)

    print("[test] Case 3: degenerate group (all Q equal)")
    # min_std=0.1 should kick in and prevent explosion
    ga3 = compute_group_advantages(
        q_values=[0.5, 0.5, 0.5, 0.5],
        r_intrinsic=[0.4, 0.7, 1.0, 0.6],   # branch 2 has best intrinsic
        r_diversity=[1.0, 0.9, 0.9, 0.9],
        min_std=0.1,
    )
    print(f"  σ_Q (clamped): {ga3.q_std:.3f}  (expect 0.1)")
    print(f"  Advantages: {[f'{a:+.3f}' for a in ga3.advantages]}")
    print(f"  gate_advantage: {ga3.gate_advantage:+.3f}  (expect 0)")
    assert ga3.q_std == 0.1
    assert abs(ga3.gate_advantage) < 1e-6
    # All q_normalized should be zero; advantages come purely from shaping
    for qn in ga3.q_normalized:
        assert abs(qn) < 1e-6

    print("[test] Case 4: shaping affects advice ordering when Q is tied")
    # Same Q for all, but different intrinsic; advice branch 2 has best intrinsic
    # Advantages should rank: branch 2 > branch 1 = branch 3 > SILENCE
    a1, a2, a3 = ga3.advantages[1], ga3.advantages[2], ga3.advantages[3]
    print(f"  Advice branch advantages: [1]={a1:+.3f} [2]={a2:+.3f} [3]={a3:+.3f}")
    assert a2 > a1  # branch 2 has higher intrinsic

    print("[test] Case 5: batch API")
    inputs = [
        ([0.2, 0.8, 0.9, 0.7], [0.4, 0.76, 1.0, 0.76], [1.0, 0.9, 0.9, 0.94]),
        ([0.9, 0.3, 0.2, 0.4], [0.4, 0.76, 1.0, 0.76], [1.0, 0.9, 0.9, 0.94]),
    ]
    batch = batch_group_advantages(inputs)
    print(f"  num_help_states: {batch.num_help_states} (expect 2)")
    print(f"  num_branch_samples: {batch.num_branch_samples} (expect 8)")
    print(f"  gate_advantages: {[f'{g:+.3f}' for g in batch.gate_advantages]}")
    assert batch.num_help_states == 2
    assert batch.num_branch_samples == 8
    assert len(batch.gate_advantages) == 2
    # Flat arrays consistent
    assert len(batch.flat_advantages) == 8
    assert len(batch.flat_state_ids) == 8
    assert len(batch.flat_branch_ids) == 8
    # Check ordering
    assert batch.flat_state_ids == [0, 0, 0, 0, 1, 1, 1, 1]
    assert batch.flat_branch_ids == [0, 1, 2, 3, 0, 1, 2, 3]

    print("[test] all cf_advantage tests passed")


if __name__ == "__main__":
    _self_test()
