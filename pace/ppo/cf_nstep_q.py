"""
cf_nstep_q.py — n-step Q with critic tail bootstrap.

Formula (method.md § 4.3):

              H_i - 1
    Q_i(s_t) =   Σ    γ^j · r_{t+j}  +  γ^{H_i} · V(s_{t+H_i})
             j=0

where:
    H_i = min(N, T_i)
    T_i = actual steps to done in branch i
    N   = max rollout steps (config-dependent)
    V   = critic prediction (0 if s is terminal)

When T_i ≤ N (branch terminated within budget): tail = 0, reduces to full MC.
When T_i > N (branch truncated): tail = γ^N · V(s_N), critic bootstraps.

We compute Q at every state along the rollout (not just s_t) because critic
loss needs targets at each visited state.

Pure numerical code — no torch/env dependency. Callers pass rewards + values
as plain Python lists / numpy arrays.

Reference: method.md § 4.3, § 5.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import math


@dataclass
class BranchTrajectory:
    """A single branch's rollout data, structured for Q computation.

    Fields:
      rewards      length H, r_{t+j} for j=0..H-1
      values       length H+1, V(s_{t+j}) for j=0..H, last one is bootstrap
                     (0 if the branch terminated; critic prediction if truncated)
      truncated    True if branch was cut at max_steps rather than reaching done
    """
    rewards: List[float]
    values: List[float]           # V at each visited state, len = len(rewards) + 1
    truncated: bool = False

    def __post_init__(self) -> None:
        if len(self.values) != len(self.rewards) + 1:
            raise ValueError(
                f"values must have len(rewards)+1 elements, "
                f"got rewards={len(self.rewards)}, values={len(self.values)}"
            )


def compute_nstep_q(
    branch: BranchTrajectory,
    gamma: float = 0.95,
) -> float:
    """Compute Q_i(s_t), i.e., the n-step return from the branch's starting state.

    Args:
      branch: rollout data starting from the state we want Q for
      gamma: discount factor

    Returns:
      Scalar Q value.
    """
    if len(branch.rewards) == 0:
        # No rollout data — Q is just the critic prediction at the start
        return float(branch.values[0])

    discount = 1.0
    q = 0.0
    for j, r in enumerate(branch.rewards):
        q += discount * float(r)
        discount *= gamma
    # Tail: γ^H · V(s_H)
    q += discount * float(branch.values[-1])
    return q


def compute_q_targets_along_trajectory(
    branch: BranchTrajectory,
    gamma: float = 0.95,
) -> List[float]:
    """Compute Q targets at EVERY state along the branch (for critic training).

    Q(s_{t+k}) = sum_{j=k..H-1} γ^{j-k} · r_{t+j}  +  γ^{H-k} · V(s_{t+H})

    Returns a list of length H+1 (one Q per visited state including the last).
    The last element equals V(s_H) itself (tail bootstrap).

    Computed backwards from the end for O(H) time.
    """
    H = len(branch.rewards)
    q_targets: List[float] = [0.0] * (H + 1)
    # Start with tail
    q_targets[H] = float(branch.values[-1])
    # Backward recursion: Q(s_k) = r_k + γ · Q(s_{k+1})
    for k in range(H - 1, -1, -1):
        q_targets[k] = float(branch.rewards[k]) + gamma * q_targets[k + 1]
    return q_targets


# ---------------------------------------------------------------------------
# Convenience helpers for typical rollout structures
# ---------------------------------------------------------------------------

def build_branch_from_rollout_data(
    rewards: Sequence[float],
    state_values: Sequence[float],
    reached_done: bool,
    critic_bootstrap_at_end: Optional[float] = None,
) -> BranchTrajectory:
    """Assemble a BranchTrajectory from raw rollout arrays.

    Args:
      rewards           length H
      state_values      length H (V at each visited state during rollout)
      reached_done      True if last step was terminal (V(terminal) = 0)
      critic_bootstrap_at_end   V(s_H) when truncated; ignored if reached_done

    Returns:
      BranchTrajectory with values length H+1.
    """
    rewards = list(rewards)
    state_values = list(state_values)
    if len(state_values) != len(rewards):
        raise ValueError(
            f"state_values must have same length as rewards, "
            f"got {len(state_values)} vs {len(rewards)}"
        )
    if reached_done:
        tail_value = 0.0
        truncated = False
    else:
        if critic_bootstrap_at_end is None:
            raise ValueError("critic_bootstrap_at_end required when not reached_done")
        tail_value = float(critic_bootstrap_at_end)
        truncated = True
    values = state_values + [tail_value]
    return BranchTrajectory(rewards=rewards, values=values, truncated=truncated)


# ---------------------------------------------------------------------------
# Sanity self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # Case 1: full MC rollout (branch reaches done)
    # rewards [0, 0.1, 0, 1.0], V at each state = [0.5, 0.4, 0.55, 0.9], reached done
    b1 = build_branch_from_rollout_data(
        rewards=[0.0, 0.1, 0.0, 1.0],
        state_values=[0.5, 0.4, 0.55, 0.9],
        reached_done=True,
    )
    q1 = compute_nstep_q(b1, gamma=0.95)
    # Expected: 0 + 0.95·0.1 + 0.95²·0 + 0.95³·1.0 + γ^4·0 = 0.095 + 0.857 = 0.952
    expected1 = 0.95 * 0.1 + (0.95 ** 3) * 1.0
    print(f"[test] Case 1 (full MC): Q = {q1:.4f} (expected {expected1:.4f})")
    assert abs(q1 - expected1) < 1e-6, "Case 1 failed"

    # Case 2: truncated rollout with tail bootstrap
    # rewards [0, 0.1], V at each state = [0.5, 0.4], truncated, V(s_end) = 0.6
    b2 = build_branch_from_rollout_data(
        rewards=[0.0, 0.1],
        state_values=[0.5, 0.4],
        reached_done=False,
        critic_bootstrap_at_end=0.6,
    )
    q2 = compute_nstep_q(b2, gamma=0.95)
    # Expected: 0 + 0.95·0.1 + 0.95²·0.6 = 0.095 + 0.5415 = 0.6365
    expected2 = 0.95 * 0.1 + (0.95 ** 2) * 0.6
    print(f"[test] Case 2 (truncated): Q = {q2:.4f} (expected {expected2:.4f})")
    assert abs(q2 - expected2) < 1e-6, "Case 2 failed"

    # Case 3: Q targets along trajectory
    q_targets = compute_q_targets_along_trajectory(b1, gamma=0.95)
    print(f"[test] Case 3 Q targets: {[f'{q:.3f}' for q in q_targets]}")
    # Last should be 0 (terminal V), first should equal compute_nstep_q result
    assert abs(q_targets[0] - q1) < 1e-6, "Case 3 first-state mismatch"
    assert abs(q_targets[-1] - 0.0) < 1e-6, "Case 3 terminal should be 0"

    # Case 4: empty rewards edge case (branch failed to step)
    b4 = BranchTrajectory(rewards=[], values=[0.5], truncated=False)
    q4 = compute_nstep_q(b4, gamma=0.95)
    print(f"[test] Case 4 (empty): Q = {q4:.4f} (expected 0.5)")
    assert abs(q4 - 0.5) < 1e-6, "Case 4 failed"

    print("[test] all n-step Q tests passed")


if __name__ == "__main__":
    _self_test()
