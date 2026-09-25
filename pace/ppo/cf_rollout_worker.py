"""
cf_rollout_worker.py — parallel main-trajectory rollout for CF-nstep-GRPO.

Extends collect_rollouts.py's rollout worker with three additions required for
PPO training (method.md § 5.5):

  1. Record token IDs and text of the companion's assistant response at every
     step (so we can re-forward the frozen "old" policy at training time to
     compute log_π_old for the PPO ratio).
  2. Call `env.snapshot()` at every HELP state and derive a Common Random
     Numbers (CRN) seed so counterfactual expansion can start from that
     exact env state under matched randomness.
  3. Apply ε-mixed gate exploration (§ 5.3): with probability
     `eps_gate_explore`, flip the sampled gate to give the scheduler
     training signal at states it would deterministically avoid HELP.

Output structure (MainTrajectory containing MainStep records) is serialized
to a Python pickle blob; the counterfactual_expander consumes it in the next
stage of the PPO pipeline.

Design reference: method.md § 4.2, § 5.3, § 5.5.
Reuses vllm client and prompt-building code from advisee.py and
prompts/alfworld.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from rl_causal.advisee import VllmAdvisee
from rl_causal.alfworld_env import ALFWorldEnv
from rl_causal.prompts.alfworld import (
    SYSTEM_PROMPT_COMPANION,
    build_companion_prompt,
)
from rl_causal.scripts.collect_rollouts import _parse_gate_advice


def _companion_prompt_bundle(env: str):
    """(system_prompt, build_companion_prompt) for the given env."""
    if env in ("minecraft", "mc"):
        from rl_causal.prompts import mc as _mcp
        return _mcp.SYSTEM_PROMPT_COMPANION, _mcp.build_companion_prompt
    return SYSTEM_PROMPT_COMPANION, build_companion_prompt


# ---------------------------------------------------------------------------
# Companion HTTP client that also returns raw response text
# ---------------------------------------------------------------------------

class VllmCompanionRaw:
    """Thin wrapper around vllm chat.completions that returns both parsed
    (gate, advice) AND the raw response text.

    Raw text is needed for training-time re-forward (tokenize the exact
    response to compute log_π_old / log_π_new for PPO ratio).
    """

    def __init__(
        self,
        model: str,
        url: str = "http://localhost:8001",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 80,      # was 200; advice rarely needs >50 tok
        timeout: float = 60.0,
        history_window: int = 10,
        env: str = "alfworld",     # "alfworld" | "minecraft"
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.history_window = history_window
        self.env = env
        self._system, self._build_prompt = _companion_prompt_bundle(env)

    def act(
        self,
        obs: dict,
        history: list,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        focus: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Return (gate, advice, raw_response_text).

        Args:
          seed         optional int for reproducible sampling
          temperature  optional override of self.temperature (used for
                       fresh CF-branch sampling where we want more diversity)
          top_p        optional override of self.top_p
          focus        optional subgoal string; anchors this advice to a
                       specific milestone (structured CF diversity)
        """
        user_prompt = self._build_prompt(
            obs, history=history, history_window=self.history_window,
            focus=focus,
        )
        msgs = [
            {"role": "system", "content": self._system},
            {"role": "user",   "content": user_prompt},
        ]
        payload = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature if temperature is None else float(temperature),
            "top_p": self.top_p if top_p is None else float(top_p),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if seed is not None:
            payload["seed"] = int(seed)
        r = requests.post(
            self.url + "/v1/chat/completions",
            json=payload, timeout=self.timeout,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        gate, advice, _ok = _parse_gate_advice(raw)
        return gate, advice, raw


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MainStep:
    """One time-step of the main trajectory during rollout collection.

    Fields used by later PPO stages:
      state_dict        — full obs dict at this step (for prompt reconstruction)
      history           — history at this step (list of (action, obs_after))
      companion_prompt  — the exact companion user prompt text used at sample time
      companion_response_text — the raw JSON text emitted by companion
      gate_sampled      — HELP or SILENCE as sampled from π_c (pre-exploration)
      gate_actual       — HELP or SILENCE after ε-mixed exploration
      advice_sampled    — advice string if gate_sampled == HELP else ""
      advice_actual     — advice string if gate_actual == HELP else "" (may be
                          freshly sampled if exploration flipped SILENCE→HELP)
      action_taken      — the advisee action executed in env
      r_env             — env-native reward
      done, truncated   — env termination flags
      snapshot          — env snapshot dict at this state if HELP else None
      crn_seed          — CRN seed derived for this state (used by branch rollouts)
      step_index        — position in main trajectory
      forced_exploration — True if gate_actual differs from gate_sampled
    """
    state_dict: Dict[str, Any]
    history: List[Tuple[str, str]]
    companion_prompt: str
    companion_response_text: str
    gate_sampled: str
    gate_actual: str
    advice_sampled: str
    advice_actual: str
    action_taken: str
    r_env: float
    done: bool
    truncated: bool
    snapshot: Optional[Dict[str, Any]]
    crn_seed: Optional[int]
    step_index: int
    forced_exploration: bool = False
    milestone_fired: bool = False   # a subtask completed at this step (resets help-cost recency)


@dataclass
class MainTrajectory:
    """A full episode from main-trajectory rollout.

    Fields:
      worker_id, episode_idx, seed: bookkeeping
      task_desc: task description
      steps: list of MainStep
      success: whether task succeeded (last r_env > 0.5 or short trial)
      trajectory_len: len(steps)
    """
    worker_id: int
    episode_idx: int
    seed: int
    task_desc: str
    steps: List[MainStep] = field(default_factory=list)
    success: bool = False
    trajectory_len: int = 0

    @property
    def help_step_indices(self) -> List[int]:
        """Indices of steps that ended up as HELP (post-exploration)."""
        return [i for i, s in enumerate(self.steps) if s.gate_actual == "HELP"]

    @property
    def silence_step_indices(self) -> List[int]:
        return [i for i, s in enumerate(self.steps) if s.gate_actual == "SILENCE"]


# ---------------------------------------------------------------------------
# CRN seed derivation
# ---------------------------------------------------------------------------

def derive_crn_seed(
    update_step: int,
    worker_id: int,
    episode_idx: int,
    step_index: int,
) -> int:
    """Deterministic CRN seed derivation.

    Same (update_step, worker_id, episode_idx, step_index) → same seed →
    same env randomness across counterfactual branches from this snapshot.
    Different steps → different seeds → decorrelated within-episode.
    """
    payload = f"{update_step}|{worker_id}|{episode_idx}|{step_index}".encode("utf-8")
    h = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(h, byteorder="big", signed=False) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# ε-mixed exploration
# ---------------------------------------------------------------------------

def _anchored_advice_fallback(state_dict: Dict[str, Any], rng: random.Random) -> str:
    """Build a real, actionable advice when a forced-exploration HELP would
    otherwise be empty.

    Anchoring to a task milestone (instead of injecting a vacuous placeholder
    like "(no specific advice)") ensures the exploratory HELP actually tests a
    *helpful* intervention. Otherwise every forced HELP measures
    "HELP with no content ≈ SILENCE", so Q_help ≈ Q_silence, the gate advantage
    stays ~0, and the gate collapses to SILENCE — a self-reinforcing artifact.
    This mirrors the milestone-anchored fresh-advice design used for the K
    counterfactual branches.
    """
    ms = state_dict.get("milestones_reference") or []
    if ms:
        pick = ms[rng.randrange(len(ms))] if len(ms) > 1 else ms[0]
        return f"Work toward this subgoal: {pick}"
    task = state_dict.get("task_description") or state_dict.get("task") or ""
    if task:
        return f"Focus on completing the task: {task}"
    cmds = state_dict.get("admissible_commands") or []
    if cmds:
        return f"Try a concrete step such as: {cmds[0]}"
    return "Take a concrete step toward the current goal."


def apply_epsilon_gate_exploration(
    gate_sampled: str,
    advice_sampled: str,
    companion: "VllmCompanionRaw",
    state_dict: Dict[str, Any],
    history: List[Tuple[str, str]],
    eps_gate_explore: float,
    rng: random.Random,
    exploration_seed: Optional[int] = None,
) -> Tuple[str, str, bool]:
    """Apply ε-mixed exploration to gate.

    With probability `eps_gate_explore`, flip HELP↔SILENCE. If the flip
    creates a HELP (from a sampled SILENCE), we need to freshly sample
    an advice from the companion.

    Returns:
      (gate_actual, advice_actual, forced_exploration_flag)
    """
    if rng.random() >= eps_gate_explore:
        return gate_sampled, advice_sampled, False

    if gate_sampled == "HELP":
        return "SILENCE", "", True
    # Flip SILENCE → HELP; need to sample fresh advice
    try:
        _, fresh_advice, _raw = companion.act(state_dict, history, seed=exploration_seed)
    except Exception:
        fresh_advice = ""
    # If the companion just returned empty/SILENCE again, DON'T inject a vacuous
    # placeholder (that makes the exploratory HELP ≈ SILENCE and collapses the
    # gate). Anchor to a task milestone so the forced HELP tests real guidance.
    if not fresh_advice:
        fresh_advice = _anchored_advice_fallback(state_dict, rng)
    return "HELP", fresh_advice, True


# ---------------------------------------------------------------------------
# One-episode rollout
# ---------------------------------------------------------------------------

def rollout_one_episode(
    env: ALFWorldEnv,
    companion: "VllmCompanionRaw",
    advisee: VllmAdvisee,
    update_step: int,
    worker_id: int,
    episode_idx: int,
    max_steps: int = 40,
    seed: Optional[int] = None,
    eps_gate_explore: float = 0.10,
    exploration_rng: Optional[random.Random] = None,
    verbose_steps: bool = True,
) -> MainTrajectory:
    """Run one full episode; return MainTrajectory."""
    if exploration_rng is None:
        exploration_rng = random.Random(seed if seed is not None else worker_id)

    t_ep_start = time.time()
    obs, info = env.reset(seed=seed)
    if verbose_steps:
        print(f"  [w{worker_id} ep{episode_idx}] reset done in {time.time()-t_ep_start:.1f}s, "
              f"task='{(obs.get('task_description','') or '')[:50]}'", flush=True)
    history: List[Tuple[str, str]] = []
    task_desc = obs.get("task_description", "") or ""

    traj = MainTrajectory(
        worker_id=worker_id,
        episode_idx=episode_idx,
        seed=seed or 0,
        task_desc=task_desc,
    )

    for t in range(max_steps):
        t_step_start = time.time() if verbose_steps else None
        # 1. Build state_dict for prompts
        state_dict = dict(obs)
        state_dict["step_count"] = t

        # 2. Build companion prompt (used later for re-forwarding at training)
        companion_prompt = build_companion_prompt(
            state_dict, history=history, history_window=companion.history_window,
        )

        # 3. Sample from companion (raw response text for later re-forward)
        try:
            gate_sampled, advice_sampled, response_text = companion.act(
                state_dict, history,
                seed=derive_crn_seed(update_step, worker_id, episode_idx, t),
            )
        except Exception as e:
            print(f"  [warn] companion.act failed at t={t}: {e}", flush=True)
            gate_sampled, advice_sampled, response_text = (
                "SILENCE", "",
                json.dumps({"gate": "SILENCE", "advice": ""}, ensure_ascii=False),
            )

        # 4. ε-mixed gate exploration
        gate_actual, advice_actual, forced = apply_epsilon_gate_exploration(
            gate_sampled, advice_sampled,
            companion, state_dict, history,
            eps_gate_explore=eps_gate_explore,
            rng=exploration_rng,
            exploration_seed=derive_crn_seed(
                update_step, worker_id, episode_idx, t * 10 + 7
            ),
        )

        # 5. Advisee decides based on gate_actual
        try:
            action = advisee.act(
                state_dict,
                advice_actual if gate_actual == "HELP" else None,
                seed=derive_crn_seed(update_step, worker_id, episode_idx, t * 10 + 3),
            )
        except Exception as e:
            print(f"  [warn] advisee.act failed at t={t}: {e}", flush=True)
            admissible = state_dict.get("admissible_commands", [])
            action = admissible[0] if admissible else "look"

        # 6. Snapshot env BEFORE stepping (state at which HELP decision was made)
        snapshot: Optional[Dict[str, Any]] = None
        crn_seed: Optional[int] = None
        if gate_actual == "HELP":
            try:
                snapshot = env.snapshot()
                crn_seed = derive_crn_seed(update_step, worker_id, episode_idx, t)
            except Exception as e:
                print(f"  [warn] env.snapshot failed at t={t}: {e}", flush=True)

        # 7. Env step
        try:
            obs_next, r_env, done, truncated, info = env.step(action)
        except ValueError:
            obs_next, r_env, done, info = env.step(action)
            truncated = False

        # 8. Record step
        traj.steps.append(MainStep(
            state_dict=state_dict,
            history=[tuple(h) for h in history],
            companion_prompt=companion_prompt,
            companion_response_text=response_text,
            gate_sampled=gate_sampled,
            gate_actual=gate_actual,
            advice_sampled=advice_sampled,
            advice_actual=advice_actual,
            action_taken=action,
            r_env=float(r_env),
            done=bool(done),
            truncated=bool(truncated),
            snapshot=snapshot,
            crn_seed=crn_seed,
            step_index=t,
            forced_exploration=forced,
            milestone_fired=bool(info.get("milestone_fired", False)) if isinstance(info, dict) else False,
        ))

        # 9. Update history + advance
        obs_next_text = obs_next.get("text", "") if isinstance(obs_next, dict) else str(obs_next)
        history.append((action, obs_next_text))
        obs = obs_next

        if verbose_steps and t_step_start is not None:
            adv_preview = (advice_actual[:200] + "...") if advice_actual and len(advice_actual) > 200 else (advice_actual or "")
            print(f"    [w{worker_id} ep{episode_idx} t{t:2d}] "
                  f"{time.time()-t_step_start:4.1f}s  gate={gate_actual:<7} "
                  f"act={action[:25]!r:<27} r={r_env:+.2f} done={done}",
                  flush=True)
            if gate_actual == "HELP" and adv_preview:
                print(f"      advice='{adv_preview}'", flush=True)

        if done or truncated:
            break

    traj.trajectory_len = len(traj.steps)
    traj.success = (
        traj.steps[-1].r_env > 0.5 if traj.steps else False
    )
    return traj


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def rollout_worker(
    worker_id: int,
    n_episodes: int,
    companion_url: str,
    companion_model: str,
    advisee_url: str,
    advisee_model: str,
    update_step: int,
    max_steps: int,
    eps_gate_explore: float,
    seed_base: int,
    companion_temperature: float = 0.7,
    companion_top_p: float = 0.95,
    excluded_task_types: Optional[set] = None,
) -> List[MainTrajectory]:
    """One worker owns one ALFWorldEnv; runs n_episodes serially."""
    try:
        env = ALFWorldEnv(excluded_task_types=excluded_task_types)
    except Exception as e:
        print(f"  [worker {worker_id}] env init failed: {e}", flush=True)
        return []

    advisee = VllmAdvisee(
        model=advisee_model, url=advisee_url,
        temperature=0.3, timeout=30.0,
    )
    companion = VllmCompanionRaw(
        model=companion_model, url=companion_url,
        temperature=companion_temperature, top_p=companion_top_p,
        timeout=30.0,
    )

    trajectories: List[MainTrajectory] = []
    exploration_rng = random.Random(seed_base + worker_id * 1000)

    for i in range(n_episodes):
        ep_seed = seed_base + worker_id * 100000 + i
        try:
            advisee.reset_history()
            traj = rollout_one_episode(
                env=env,
                companion=companion,
                advisee=advisee,
                update_step=update_step,
                worker_id=worker_id,
                episode_idx=i,
                max_steps=max_steps,
                seed=ep_seed,
                eps_gate_explore=eps_gate_explore,
                exploration_rng=exploration_rng,
            )
            trajectories.append(traj)
            print(
                f"  [worker {worker_id}] episode {i+1}/{n_episodes}  "
                f"len={traj.trajectory_len}  success={traj.success}  "
                f"help_states={len(traj.help_step_indices)}  "
                f"forced={sum(1 for s in traj.steps if s.forced_exploration)}",
                flush=True,
            )
        except Exception as e:
            # ALFWorld env internal state can get corrupted after some
            # episodes (esp. after successful termination or many snapshots).
            # Print traceback for diagnosis, then rebuild env to recover.
            import traceback
            print(f"  [worker {worker_id}] episode {i} failed: {type(e).__name__}: {e}",
                  flush=True)
            print(f"  [worker {worker_id}] traceback:\n{traceback.format_exc()}",
                  flush=True)
            print(f"  [worker {worker_id}] rebuilding env to recover...", flush=True)
            try:
                env = ALFWorldEnv(excluded_task_types=excluded_task_types)
            except Exception as e2:
                print(f"  [worker {worker_id}] env rebuild failed: {e2}", flush=True)
                break

    return trajectories


# ---------------------------------------------------------------------------
# CLI entry (smoke test)
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--eps-gate-explore", type=float, default=0.10)
    ap.add_argument("--update-step", type=int, default=0,
                    help="PPO update step id (for CRN seed reproducibility)")
    ap.add_argument("--companion-url", default="http://localhost:8001")
    ap.add_argument("--companion-model", default="companion")
    ap.add_argument("--advisee-url", default="http://localhost:8000")
    ap.add_argument("--advisee-model", default="/workspace/models/Qwen3.5-4B")
    ap.add_argument("--companion-temperature", type=float, default=0.7)
    ap.add_argument("--companion-top-p", type=float, default=0.95)
    ap.add_argument("--output", required=True,
                    help="Output pickle path for MainTrajectory list")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n_per_worker = max(1, args.n_episodes // args.parallel)
    workers_ep = [n_per_worker] * args.parallel
    for i in range(args.n_episodes - n_per_worker * args.parallel):
        workers_ep[i] += 1

    print("=" * 70)
    print("cf_rollout_worker — main trajectory collection")
    print("=" * 70)
    print(f"  n_episodes:       {args.n_episodes}")
    print(f"  parallel:         {args.parallel}")
    print(f"  max_steps:        {args.max_steps}")
    print(f"  eps_gate_explore: {args.eps_gate_explore}")
    print(f"  update_step:      {args.update_step}")
    print(f"  output:           {args.output}")
    print("=" * 70, flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    all_trajectories: List[MainTrajectory] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                rollout_worker,
                worker_id=w, n_episodes=workers_ep[w],
                companion_url=args.companion_url, companion_model=args.companion_model,
                advisee_url=args.advisee_url, advisee_model=args.advisee_model,
                update_step=args.update_step,
                max_steps=args.max_steps,
                eps_gate_explore=args.eps_gate_explore,
                seed_base=args.seed + w * 10000,
                companion_temperature=args.companion_temperature,
                companion_top_p=args.companion_top_p,
            ): w
            for w in range(args.parallel)
        }
        for fut in as_completed(futures):
            worker_id = futures[fut]
            try:
                trajs = fut.result()
                all_trajectories.extend(trajs)
                dt = time.time() - t_start
                print(f"  [progress] worker {worker_id} done, "
                      f"{len(all_trajectories)} trajs total, {dt:.0f}s elapsed",
                      flush=True)
            except Exception as e:
                print(f"  [error] worker {worker_id} failed: {e}", flush=True)

    # Serialize
    with open(args.output, "wb") as fout:
        pickle.dump(all_trajectories, fout)

    # Summary
    dt = time.time() - t_start
    n_steps = sum(t.trajectory_len for t in all_trajectories)
    n_help = sum(len(t.help_step_indices) for t in all_trajectories)
    n_success = sum(1 for t in all_trajectories if t.success)
    n_forced = sum(
        sum(1 for s in t.steps if s.forced_exploration) for t in all_trajectories
    )

    print()
    print("=" * 70)
    print(f"[done] wrote {len(all_trajectories)} trajectories to {args.output}")
    print(f"  time            : {dt:.0f}s ({dt/60:.1f} min)")
    print(f"  total steps     : {n_steps}")
    print(f"  HELP states     : {n_help}")
    print(f"  forced explored : {n_forced}")
    print(f"  success rate    : {n_success}/{len(all_trajectories)} "
          f"({100*n_success/max(1,len(all_trajectories)):.1f}%)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
