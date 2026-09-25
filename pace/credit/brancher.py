"""
brancher.py — counterfactual sampling at a replay node.

Given a snapshotted env state (a "node"), sample K_advices × K_seed branches
to estimate the causal value of intervention. See docs/interfaces.md § 2 and
method.md § 4.3.

Core contract:
  - Every branch starts by env.restore(node_snapshot) + env.set_rng_state(seed)
  - HELP and SILENCE branches at the same node SHARE the seed set (CRN)
  - CRN lets Δ = R(HELP) - R(SILENCE) cancel advisee-internal noise, so the
    remaining Δ is (up to fidelity residuals) the pure causal effect of the
    intervention. See experiment_plan.md Appendix A.

Three main use cases (interfaces.md § 2.4):
  - E3 ruler ground truth: force_decision ∈ {HELP, SILENCE}, K_advices=K
  - PC2 Δ-distribution: same, sweep across many nodes
  - DR training samples: force_decision=None (walk ε-mixed policy), K_advices=1
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BranchResult:
    """Aggregated result for one (decision, advice) branch replicated across
    K_seed CRN seeds. See interfaces.md § 2.1."""
    decision: str                       # "HELP" | "SILENCE"
    advice: Optional[str]               # None if SILENCE
    seed_returns: list                  # G per CRN seed
    milestones_fired: list              # per-seed list[str] of milestone labels
    steps_to_terminal: list             # per-seed step count (int)
    won_per_seed: list                  # per-seed bool
    advice_source: str = "sampled"      # "sampled" | "forced" | "placebo" | "silent"
    node_id: str = ""
    seeds: list = field(default_factory=list)

    @property
    def mean_return(self) -> float:
        return (sum(self.seed_returns) / len(self.seed_returns)) if self.seed_returns else 0.0

    @property
    def win_rate(self) -> float:
        return (sum(self.won_per_seed) / len(self.won_per_seed)) if self.won_per_seed else 0.0


# ---------------------------------------------------------------------------
# Interfaces (Protocols — duck-typed, mock-friendly)
# ---------------------------------------------------------------------------

@runtime_checkable
class AdviseeInterface(Protocol):
    """Frozen advisee LLM: (obs, advice, seed) → action string.

    Determinism contract: same (obs, advice, seed) MUST return same action.
    PC1-seedable validates this against the underlying vllm/openai model."""
    def act(self, obs: dict, advice: Optional[str], seed: int) -> str: ...


@runtime_checkable
class CompanionInterface(Protocol):
    """Companion policy — decides advice content."""
    def sample_advice(self, obs: dict, seed: int) -> str: ...


@runtime_checkable
class SnapshottableEnv(Protocol):
    """The env contract brancher relies on. ALFWorldEnv + MockALFWorldEnv
    both satisfy this."""
    def restore(self, snap: bytes) -> None: ...
    def step(self, action: str) -> tuple: ...
    def rng_state(self) -> bytes: ...
    def set_rng_state(self, state: bytes) -> None: ...
    def get_current_obs(self) -> dict: ...
    @property
    def admissible_commands(self) -> list: ...


# ---------------------------------------------------------------------------
# Seed derivation (CRN foundation)
# ---------------------------------------------------------------------------

def derive_seeds(node_id: str, K_seed: int, seed_base: int = 0) -> list:
    """Deterministic seed list keyed on node_id. Same node → same seeds.
    HELP and SILENCE at the same node reuse this list → CRN."""
    seeds = []
    for k in range(K_seed):
        h = hashlib.sha1(f"{node_id}::{seed_base}::{k}".encode("utf-8")).digest()
        seeds.append(int.from_bytes(h[:4], "little") % (2 ** 31 - 1))
    return seeds


def _step_seed(base_seed: int, step_i: int) -> int:
    """Derive a per-step seed from (base_seed, step_i) via hash.

    Replaces the naive `base_seed + step_i` linear scheme — that had a
    theoretical collision risk when two nodes have base seeds within
    max_steps of each other. Hash-based derivation makes each step's seed
    independent of the specific (base, step) split.
    """
    h = hashlib.sha1(f"{base_seed}::{step_i}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % (2 ** 31 - 1)


def _seed_to_rng_state(seed: int) -> bytes:
    """Convert int seed → env-compatible pickled RNG state."""
    import random as _r
    _r.seed(seed)
    state = {"py": _r.getstate()}
    try:
        import numpy as _np
        _np.random.seed(seed)
        state["np"] = _np.random.get_state()
    except ImportError:
        state["np"] = None
    return pickle.dumps(state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from rl_causal.utils import unwrap_won as _unwrap_won  # noqa: E402


# ---------------------------------------------------------------------------
# Single-branch rollout
# ---------------------------------------------------------------------------

def _rollout_branch(
    env: SnapshottableEnv,
    advisee: AdviseeInterface,
    node_snapshot: bytes,
    seed: int,
    advice: Optional[str],
    max_steps: int = 50,
    reward_fn: Optional[Callable] = None,
) -> dict:
    """Restore env → seed it → roll out under fixed (advice, seed).

    Returns dict: {"return", "steps", "milestones", "won"}.

    reward_fn signature (optional): (obs, action, r_env, done, info) → float.
    Defaults to env-native reward.
    """
    env.restore(node_snapshot)
    env.set_rng_state(_seed_to_rng_state(seed))
    # Advisees with per-episode history (e.g. VllmAdvisee) need reset between
    # branches so HELP and SILENCE prompts don't leak history from each other.
    if hasattr(advisee, "reset_history"):
        advisee.reset_history()

    total_return = 0.0
    milestones_all: list = []
    won = False
    obs_dict = env.get_current_obs()
    steps_taken = 0

    import os as _os
    _dbg = bool(_os.environ.get("BRANCH_DEBUG"))
    for step_i in range(max_steps):
        if _dbg and step_i < 6:
            _adm = obs_dict.get("admissible_commands") if isinstance(obs_dict, dict) else None
            _txt = (obs_dict.get("text", "") if isinstance(obs_dict, dict) else str(obs_dict))[:100]
            print(f"      [dbg t={step_i}] n_adm={len(_adm) if _adm else 0} "
                  f"advice={'Y' if advice else 'N'} obs={_txt!r}", flush=True)
        action = advisee.act(obs_dict, advice, _step_seed(seed, step_i))
        obs_dict, r_env, done, _trunc, info = env.step(action)
        steps_taken = step_i + 1
        if _dbg and step_i < 6:
            print(f"      [dbg t={step_i}] act={action!r} r_env={r_env} done={done} "
                  f"won?={_unwrap_won(info)}", flush=True)

        r = reward_fn(obs_dict, action, r_env, done, info) if reward_fn else float(r_env)
        total_return += r

        fired = env.check_milestones(info) if hasattr(env, "check_milestones") else []
        if fired:
            milestones_all.extend(fired)

        if done:
            won = _unwrap_won(info) or (float(r_env) > 0)
            break

    return {
        "return": total_return,
        "steps": steps_taken,
        "milestones": milestones_all,
        "won": won,
    }


# ---------------------------------------------------------------------------
# Branch planning
# ---------------------------------------------------------------------------

def _plan_branches(
    companion: Optional[CompanionInterface],
    initial_obs: dict,
    force_decision: Optional[str],
    force_advice: Optional[str],
    K_advices: int,
    placebo: bool,
) -> list:
    """Enumerate list of (decision, advice, source) to run.

    Cases:
      - force_decision="SILENCE" → [(SILENCE, None, "silent")]
      - force_decision="HELP" + force_advice → [(HELP, force_advice, "forced")]
      - force_decision="HELP" + no force_advice → K_advices sampled from companion
      - force_decision=None + companion → 1 SILENCE + K_advices HELP (mixed default)
      - force_decision=None + no companion → 1 SILENCE only (nothing to sample)
      - placebo=True → also append (HELP, "", "placebo")
    """
    plan: list = []

    if force_decision == "SILENCE":
        plan.append(("SILENCE", None, "silent"))
        return plan

    if force_decision == "HELP":
        if force_advice is not None:
            plan.append(("HELP", force_advice, "forced"))
        else:
            if companion is None:
                raise ValueError(
                    "force_decision='HELP' without force_advice requires a companion"
                )
            for k in range(K_advices):
                plan.append(("HELP", companion.sample_advice(initial_obs, seed=k), "sampled"))
        if placebo:
            plan.append(("HELP", "", "placebo"))
        return plan

    # force_decision is None → mixed default
    plan.append(("SILENCE", None, "silent"))
    if companion is not None:
        for k in range(K_advices):
            plan.append(("HELP", companion.sample_advice(initial_obs, seed=k), "sampled"))
    if placebo:
        plan.append(("HELP", "", "placebo"))
    return plan


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def branch_at_node(
    env: SnapshottableEnv,
    advisee: AdviseeInterface,
    companion: Optional[CompanionInterface],
    node_snapshot: bytes,
    node_id: str = "",
    force_decision: Optional[str] = None,
    force_advice: Optional[str] = None,
    K_advices: int = 3,
    K_seed: int = 4,
    placebo: bool = False,
    max_steps: int = 50,
    reward_fn: Optional[Callable] = None,
    seed_base: int = 0,
) -> list:
    """Sample counterfactual branches from a replay node.

    Returns: list[BranchResult] — one per (decision, advice) combo.
    Each BranchResult aggregates K_seed CRN-seed replicates.

    Preconditions:
      - env supports restore / set_rng_state / step / get_current_obs
      - node_snapshot was produced by env.snapshot()
      - advisee is deterministic under fixed (obs, advice, seed) — PC1-seedable

    Postconditions:
      - Every branch uses the SAME K_seed seed set derived from node_id (CRN)
      - env is left in some post-branch state — caller should not assume the
        env is at the node afterward

    See interfaces.md § 2.4 for typical use cases.
    """
    # 1. Peek current obs (for companion advice sampling), then re-restore
    env.restore(node_snapshot)
    initial_obs = env.get_current_obs()

    # 2. Enumerate branches
    plan = _plan_branches(
        companion=companion, initial_obs=initial_obs,
        force_decision=force_decision, force_advice=force_advice,
        K_advices=K_advices, placebo=placebo,
    )

    # 3. CRN seeds — shared across all branches at this node
    seeds = derive_seeds(node_id, K_seed, seed_base=seed_base)

    # 4. Execute
    results: list = []
    for decision, advice, source in plan:
        effective_advice = advice if decision == "HELP" else None
        seed_returns, seed_ms, seed_steps, seed_won = [], [], [], []
        for s in seeds:
            r = _rollout_branch(
                env=env, advisee=advisee, node_snapshot=node_snapshot,
                seed=s, advice=effective_advice,
                max_steps=max_steps, reward_fn=reward_fn,
            )
            seed_returns.append(r["return"])
            seed_ms.append(r["milestones"])
            seed_steps.append(r["steps"])
            seed_won.append(r["won"])
        results.append(BranchResult(
            decision=decision,
            advice=advice,
            seed_returns=seed_returns,
            milestones_fired=seed_ms,
            steps_to_terminal=seed_steps,
            won_per_seed=seed_won,
            advice_source=source,
            node_id=node_id,
            seeds=list(seeds),
        ))

    return results


# ---------------------------------------------------------------------------
# Convenience aggregations
# ---------------------------------------------------------------------------

def delta_help_minus_silence(results: list) -> Optional[float]:
    """Δ = mean(R_HELP) − mean(R_SILENCE) across a list of BranchResults.

    Averages across all HELP branches (any advice) and the SILENCE branch.
    Returns None if either side is missing."""
    help_returns, silence_returns = [], []
    for br in results:
        if br.decision == "HELP":
            help_returns.extend(br.seed_returns)
        elif br.decision == "SILENCE":
            silence_returns.extend(br.seed_returns)
    if not help_returns or not silence_returns:
        return None
    return (sum(help_returns) / len(help_returns)) - (sum(silence_returns) / len(silence_returns))
