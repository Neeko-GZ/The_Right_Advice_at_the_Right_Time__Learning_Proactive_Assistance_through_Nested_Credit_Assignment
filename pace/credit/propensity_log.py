"""
propensity_log.py — per-step trajectory logging with propensities.

Every rollout step produces a `Transition` dict; PropensityLogger appends
one JSON line per step so the DR estimator can compute propensity-weighted
corrections offline.

Schema follows `docs/interfaces.md § 3`. All rewards / probabilities are
stored per-step (not per-episode) so that off-policy correction can be
done at any granularity.

Consistency invariants (checked in tests):
    p_help_behavior == (1 − ε) · p_help_policy + ε · 0.5
    r_total         == r_milestone + r_terminal + r_cost + r_length
    is_explore ⇒ decision differs from argmax(p_help_policy)  (probabilistic)

Usage:

    logger = PropensityLogger("data/rollout_20260710.jsonl")
    for step in episode:
        tx = Transition(...)
        logger.append(tx)
    logger.close()   # or context-manager

Reading back for DR:

    for tx in PropensityLogger.iter_read("data/rollout_20260710.jsonl"):
        ...
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator, Literal, Optional


DecisionT = Literal["HELP", "SILENCE"]


# ---------------------------------------------------------------------------
# per-step schema
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """One rollout step, everything needed for DR / PPO / audit.

    All floats stored as-is (no rounding); readers may downcast for
    memory. Missing / not-applicable fields carry sensible zeros
    rather than nulls, except `advice*` and `node_ckpt_id` which are
    Optional and may be None.
    """

    # ── identity ────────────────────────────────────────────────────
    episode_id: str          # uuid or "seedX_taskY_epZ"
    step: int                # 0-indexed within episode
    task_name: str           # from task_pool
    state_hash: str          # deterministic hash of obs (for grouping)

    # ── action ──────────────────────────────────────────────────────
    decision: DecisionT
    advice: Optional[str] = None
    advice_ids: Optional[list[int]] = None   # token ids for PPO log-p replay

    # ── propensity (DR requires exact) ──────────────────────────────
    p_help_policy: float = 0.5      # π_θ(HELP|s), un-mixed
    p_help_behavior: float = 0.5    # π_b(HELP|s) = (1−ε)·p_help_policy + ε·0.5
    epsilon: float = 0.0            # current ε in behavior-mixing
    is_explore: bool = False        # this step's decision came from ε-branch

    # ── reward breakdown ────────────────────────────────────────────
    r_milestone: float = 0.0
    r_terminal: float = 0.0
    r_cost: float = 0.0
    r_length: float = 0.0
    r_total: float = 0.0

    # ── milestone bookkeeping ───────────────────────────────────────
    milestones_fired: list[str] = field(default_factory=list)

    # ── replay-node linkage (optional) ──────────────────────────────
    node_ckpt_id: Optional[str] = None
    is_verified_node: bool = False

    # ── time ────────────────────────────────────────────────────────
    wall_time: float = field(default_factory=lambda: time.time())

    # ── serialization ───────────────────────────────────────────────

    def to_json(self) -> str:
        """One-line JSON serialization."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "Transition":
        d = json.loads(line)
        return cls(**d)


# ---------------------------------------------------------------------------
# logger
# ---------------------------------------------------------------------------

class PropensityLogger:
    """Append-only JSONL writer for rollout transitions.

    Not thread-safe — one instance per rollout worker. If multi-worker
    logging is needed, either use one file per worker or add a mutex.

    Flushes to disk on `flush()` or `close()`; between those, writes go
    into the OS buffer.
    """

    def __init__(self, output_path: str | os.PathLike, *, buffered: bool = True) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered if not buffered; block-buffered otherwise
        self._fh = self.output_path.open("a", buffering=(1 if not buffered else -1))
        self._count = 0

    def append(self, tx: Transition) -> None:
        self._fh.write(tx.to_json() + "\n")
        self._count += 1

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "PropensityLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def count(self) -> int:
        return self._count

    # ── read-back ───────────────────────────────────────────────────

    @staticmethod
    def iter_read(path: str | os.PathLike) -> Iterator[Transition]:
        """Stream transitions from a jsonl file (line by line, low memory)."""
        with Path(path).open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield Transition.from_json(line)


# ---------------------------------------------------------------------------
# convenience: derive p_help_behavior from ε-mixing
# ---------------------------------------------------------------------------

def epsilon_mix(p_help_policy: float, epsilon: float) -> float:
    """π_b(HELP|s) = (1 − ε) · π_θ(HELP|s) + ε · 0.5.

    ε ∈ [0, 1]. When ε=0, behavior == policy. When ε=1, uniform random.
    """
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    return (1.0 - epsilon) * float(p_help_policy) + epsilon * 0.5


# ---------------------------------------------------------------------------
# consistency helpers (used by tests + optional runtime checks)
# ---------------------------------------------------------------------------

def check_epsilon_mix_consistency(tx: Transition, tol: float = 1e-6) -> bool:
    expected = epsilon_mix(tx.p_help_policy, tx.epsilon)
    return abs(tx.p_help_behavior - expected) <= tol


def check_reward_sum_consistency(tx: Transition, tol: float = 1e-6) -> bool:
    expected = tx.r_milestone + tx.r_terminal + tx.r_cost + tx.r_length
    return abs(tx.r_total - expected) <= tol


def check_all(tx: Transition, tol: float = 1e-6) -> tuple[bool, list[str]]:
    """Run all consistency checks; return (all_ok, list_of_failing_checks)."""
    failing = []
    if not check_epsilon_mix_consistency(tx, tol):
        failing.append("epsilon_mix")
    if not check_reward_sum_consistency(tx, tol):
        failing.append("reward_sum")
    return (len(failing) == 0), failing


# ---------------------------------------------------------------------------
# tiny id helper (episode_id when caller doesn't supply one)
# ---------------------------------------------------------------------------

def make_episode_id(seed: Optional[int] = None,
                    task: Optional[str] = None,
                    ep_idx: Optional[int] = None) -> str:
    """Structured episode id: `seed{seed}_{task}_ep{ep_idx}` if all given,
    else uuid4."""
    if seed is not None and task is not None and ep_idx is not None:
        return f"seed{seed}_{task}_ep{ep_idx}"
    return uuid.uuid4().hex[:16]
