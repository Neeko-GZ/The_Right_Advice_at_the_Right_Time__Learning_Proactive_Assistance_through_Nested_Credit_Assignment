"""
cf_expander.py — K+2 branch counterfactual expansion per HELP state.

For each HELP state s_t in a main trajectory, roll out K+2 branches:

  1. SILENCE branch    — brancher.restore(snap) + seed CRN + force gate=SILENCE,
                         rollout up to N steps.
  2. HELP-replay branch — brancher.restore(snap) + seed CRN + force gate=HELP
                         + force main-trajectory's advice, rollout up to N steps.
                         Required for a clean gate counterfactual (§ 4.2): the
                         main trajectory's own rollout used a different env
                         seed, so its return is not apples-to-apples with the
                         SILENCE branch.
  3. K advice branches  — sample K fresh advice from companion, force each,
                         rollout up to N steps. Fresh advice = π_c samples
                         excluding the main trajectory's one (which is the
                         HELP-replay branch).

All K+2 branches:
  * start from the same env snapshot (state alignment)
  * share the CRN seed (advisee seed + env noise alignment)
  * cap at max_steps N (truncation for MC-friendly cost)

Output: an ExpandedHelpState record per HELP state, containing the
per-branch rollout data (rewards + visited states) that
`cf_nstep_q.compute_nstep_q` and `cf_advantage.*` consume.

Design reference: method.md § 4.2, § 4.3, § 5.5.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from rl_causal.advisee import VllmAdvisee
from rl_causal.alfworld_env import ALFWorldEnv
from rl_causal.prompts.alfworld import build_companion_prompt

from rl_causal.ppo.cf_rollout_worker import (
    MainStep,
    MainTrajectory,
    VllmCompanionRaw,
    derive_crn_seed,
)


# ---------------------------------------------------------------------------
# Branch data structures
# ---------------------------------------------------------------------------

@dataclass
class BranchRollout:
    """One counterfactual branch's rollout data.

    Fields:
      branch_type       — "silence" | "help_replay" | "advice"
      advice_text       — the advice that was forced (empty for silence)
      advice_response_text — raw JSON emitted by the companion sample call
                             that produced this advice (None for silence
                             and help_replay, since help_replay reuses main's
                             advice which is already tokenized elsewhere)
      rewards           — env rewards per step in the branch (length H)
      state_prompts     — companion state prompts for each visited state
                          (length H+1, for critic re-forward)
      last_obs_text     — the observation text at the last visited state
                          (for critic V(s_H) computation)
      reached_done      — True if branch reached env done
      truncated         — True if branch was cut at max_steps
      length            — H (number of steps actually run)
    """
    branch_type: str
    advice_text: str
    advice_response_text: Optional[str]
    rewards: List[float]
    state_prompts: List[str]         # companion prompts for each visited state
    last_obs_text: str
    reached_done: bool
    truncated: bool
    length: int
    gates: List[str] = field(default_factory=list)   # per-step gate "HELP"/"SILENCE" (len H)
    milestone_fired: List[bool] = field(default_factory=list)  # subtask done at step (len H)


@dataclass
class ExpandedHelpState:
    """All K+2 branches rolled out for a single HELP state.

    Fields:
      main_trajectory_id   — (worker_id, episode_idx) of the parent traj
      step_index           — position in main trajectory
      snapshot_used        — the env snapshot dict (kept for reproducibility)
      crn_seed             — CRN seed shared by all K+2 branches

      silence_branch       — the SILENCE branch
      help_replay_branch   — the HELP-replay branch (main advice)
      advice_branches      — list of K BranchRollouts (fresh advice)

      main_advice_text     — advice used in main trajectory at this state
    """
    main_trajectory_id: Tuple[int, int]
    step_index: int
    snapshot_used: Any                    # bytes (pickled) for ALFWorld
    crn_seed: int
    silence_branch: BranchRollout
    help_replay_branch: BranchRollout
    advice_branches: List[BranchRollout]
    main_advice_text: str


# ---------------------------------------------------------------------------
# Core: rollout a single branch from a snapshot under forced action
# ---------------------------------------------------------------------------

def rollout_branch(
    env: ALFWorldEnv,
    advisee: VllmAdvisee,
    companion: VllmCompanionRaw,
    snapshot: Any,                     # bytes (pickle) or dict, opaque to us
    crn_seed: int,
    forced_gate: str,
    forced_advice: str,
    max_steps: int,
    branch_type: str,
    initial_obs: Dict[str, Any],        # obs at the snapshot state
    initial_history: List[Tuple[str, str]],  # history at the snapshot state
    advice_response_text: Optional[str] = None,
) -> BranchRollout:
    """Restore env to snapshot, force the first action per (forced_gate,
    forced_advice), then let policy sample naturally for subsequent steps.

    "Force" means: at t=t_0 (the snapshot's state), use the given
    (gate, advice) to determine advisee action; at t > t_0, sample gate
    and advice from the companion naturally (no counterfactual re-expansion).

    Args:
      env               — ALFWorldEnv instance
      advisee           — VllmAdvisee client
      companion         — VllmCompanionRaw client (for post-first-step samples)
      snapshot          — opaque env.snapshot() blob (bytes for ALFWorld)
      crn_seed          — CRN seed for reproducibility
      forced_gate       — "HELP" or "SILENCE" for the first branch step
      forced_advice     — advice string if forced_gate == "HELP"
      max_steps         — truncation budget N
      branch_type       — "silence" | "help_replay" | "advice" for record
      initial_obs       — obs at the snapshotted state (from MainStep.state_dict)
      initial_history   — history at the snapshotted state (from MainStep.history)
      advice_response_text — raw JSON emitted for the branch's advice (only
                             passed for fresh-advice branches)

    Returns:
      BranchRollout with rewards + state_prompts + last_obs_text.
    """
    # Restore + seed
    try:
        env.restore(snapshot)
    except Exception as e:
        # If restore fails, return a degenerate branch (0-length)
        return BranchRollout(
            branch_type=branch_type,
            advice_text=forced_advice if forced_gate == "HELP" else "",
            advice_response_text=advice_response_text,
            rewards=[],
            state_prompts=[""],
            last_obs_text=f"[restore_failed:{e}]",
            reached_done=False,
            truncated=False,
            length=0,
        )
    try:
        env.set_seed(crn_seed)
    except AttributeError:
        # Some envs don't have set_seed; ok to skip
        pass

    # Advisee reset per branch (fresh history)
    try:
        advisee.reset_history()
    except AttributeError:
        pass

    # Use obs + history from the MainStep record (we can't peek into
    # the opaque snapshot; it's bytes for ALFWorld).
    obs = dict(initial_obs)
    history: List[Tuple[str, str]] = [tuple(h) for h in initial_history]

    rewards: List[float] = []
    gates: List[str] = []
    ms_fired: List[bool] = []
    state_prompts: List[str] = []
    reached_done = False
    truncated = False

    for t in range(max_steps):
        # Build state dict for prompts
        state_dict = dict(obs) if isinstance(obs, dict) else {"text": str(obs)}
        state_dict["step_count"] = t

        # Record the state prompt for critic V(s) re-forward later
        state_prompts.append(build_companion_prompt(
            state_dict, history=history, history_window=10,
        ))

        # Decide gate + advice for this step
        if t == 0:
            gate_used, advice_used = forced_gate, forced_advice
        else:
            # After the first step, sample naturally from companion (no
            # re-expansion into further counterfactual branches).
            try:
                gate_used, advice_used, _raw = companion.act(
                    state_dict, history,
                    seed=crn_seed + t * 17,
                )
            except Exception as e:
                gate_used, advice_used = "SILENCE", ""

        # Advisee decides
        try:
            action = advisee.act(
                state_dict,
                advice_used if gate_used == "HELP" else None,
                seed=crn_seed + t * 31 + 3,
            )
        except Exception as e:
            admissible = state_dict.get("admissible_commands", [])
            action = admissible[0] if admissible else "look"

        # Step env
        try:
            obs_next, r_env, done, truncated_flag, info = env.step(action)
        except ValueError:
            obs_next, r_env, done, info = env.step(action)
            truncated_flag = False
        except Exception as e:
            # env failure — stop branch
            break

        rewards.append(float(r_env))
        gates.append(gate_used)
        ms_fired.append(bool(info.get("milestone_fired", False)) if isinstance(info, dict) else False)

        # Update history and obs
        obs_next_text = obs_next.get("text", "") if isinstance(obs_next, dict) else str(obs_next)
        history.append((action, obs_next_text))
        obs = obs_next

        if done:
            reached_done = True
            break
        if truncated_flag:
            truncated = True
            break

    # Record final state's prompt (for V(s_H))
    final_state_dict = dict(obs) if isinstance(obs, dict) else {"text": str(obs)}
    final_state_dict["step_count"] = len(rewards)
    state_prompts.append(build_companion_prompt(
        final_state_dict, history=history, history_window=10,
    ))

    if len(rewards) >= max_steps and not reached_done:
        truncated = True

    last_obs_text = obs.get("text", "") if isinstance(obs, dict) else str(obs)

    return BranchRollout(
        branch_type=branch_type,
        advice_text=forced_advice if forced_gate == "HELP" else "",
        advice_response_text=advice_response_text,
        rewards=rewards,
        state_prompts=state_prompts,
        last_obs_text=last_obs_text,
        reached_done=reached_done,
        truncated=truncated,
        length=len(rewards),
        gates=gates,
        milestone_fired=ms_fired,
    )


# ---------------------------------------------------------------------------
# Expand one HELP state
# ---------------------------------------------------------------------------

def expand_help_state(
    env: ALFWorldEnv,
    advisee: VllmAdvisee,
    companion: VllmCompanionRaw,
    main_trajectory: MainTrajectory,
    step_index: int,
    K: int,
    max_steps: int,
    fresh_temperature: float = 0.8,
    fresh_top_p: float = 0.98,
    debug: bool = False,
) -> Optional[ExpandedHelpState]:
    """Run K+2 branches for a single HELP state.

    Args:
      fresh_temperature — temperature used when sampling K fresh advice
                          from companion. Higher than main-traj temperature
                          (typically 0.7) to encourage diversity across
                          counterfactual branches; matches the method-level
                          intent of exploring different advice options.
      fresh_top_p       — top-p used for fresh sampling.
      debug             — if True, print each fresh sample and any fallback

    Returns None if the step is not a HELP state or the snapshot is missing.
    """
    step: MainStep = main_trajectory.steps[step_index]
    if step.gate_actual != "HELP" or step.snapshot is None or step.crn_seed is None:
        return None

    # Recover the obs + history that were present when we snapshotted.
    # (Snapshot is opaque bytes for ALFWorld; we can't peek in.)
    initial_obs = step.state_dict if isinstance(step.state_dict, dict) else {"text": str(step.state_dict)}
    initial_history = list(step.history) if step.history else []

    # ABLATION 3c (GRPO): sample G = K+2 FREE full {gate,advice} outputs from the
    # companion and roll each under its OWN sampled gate (HELP -> give advice,
    # SILENCE -> no advice). No forced silence/replay anchor, no milestone focus;
    # all G branches form the group. NIC_EXPAND_MODE=grpo activates this.
    if os.environ.get("NIC_EXPAND_MODE", "").lower() == "grpo":
        G = max(2, K + 2)
        free_branches: List[BranchRollout] = []
        for k in range(G):
            seed_k = step.crn_seed + 101 * (k + 1)
            try:
                g_gate, g_advice, g_raw = companion.act(
                    obs=step.state_dict, history=step.history,
                    seed=seed_k, temperature=fresh_temperature,
                    top_p=fresh_top_p, focus=None,
                )
            except Exception:
                g_gate, g_advice, g_raw = "SILENCE", "", ""
            is_help = str(g_gate).upper() == "HELP"
            br = rollout_branch(
                env=env, advisee=advisee, companion=companion,
                snapshot=step.snapshot, crn_seed=step.crn_seed,
                forced_gate=("HELP" if is_help else "SILENCE"),
                forced_advice=(g_advice if is_help else ""),
                max_steps=max_steps, branch_type="advice",
                initial_obs=initial_obs, initial_history=initial_history,
                advice_response_text=g_raw,
            )
            free_branches.append(br)
        return ExpandedHelpState(
            main_trajectory_id=(main_trajectory.worker_id, main_trajectory.episode_idx),
            step_index=step_index, snapshot_used=step.snapshot,
            crn_seed=step.crn_seed,
            silence_branch=free_branches[0],
            help_replay_branch=free_branches[1],
            advice_branches=free_branches[2:],
            main_advice_text=step.advice_actual,
        )

    # 1. SILENCE branch
    silence_branch = rollout_branch(
        env=env, advisee=advisee, companion=companion,
        snapshot=step.snapshot,
        crn_seed=step.crn_seed,
        forced_gate="SILENCE",
        forced_advice="",
        max_steps=max_steps,
        branch_type="silence",
        initial_obs=initial_obs,
        initial_history=initial_history,
    )

    # 2. HELP-replay branch (use main's advice, but under CRN)
    help_replay_branch = rollout_branch(
        env=env, advisee=advisee, companion=companion,
        snapshot=step.snapshot,
        crn_seed=step.crn_seed,
        forced_gate="HELP",
        forced_advice=step.advice_actual,
        max_steps=max_steps,
        branch_type="help_replay",
        initial_obs=initial_obs,
        initial_history=initial_history,
        advice_response_text=step.companion_response_text,
    )

    # 3. K advice branches. Structured (milestone-anchored) diversity: instead
    # of relying on high temperature (which produces DIVERSE but HALLUCINATED
    # advice — the K samples then all fail and carry no content signal), we
    # sample at a moderate temperature and anchor each branch to a DIFFERENT
    # subgoal from the reference milestones. This yields advice that is both
    # coherent and genuinely different in content → real Q spread → usable
    # content credit. Env-agnostic: milestones exist in ALFWorld and Minecraft.
    milestones = (step.state_dict.get("milestones_reference", [])
                  if isinstance(step.state_dict, dict) else []) or []
    # NIC_FOCUS_MODE=horizon -> branch k anchors to the next (k+1) milestones as an
    # ordered plan (horizon diversity: some branches short, some longer, and content
    # credit reinforces whichever depth helps). Default "index" = one milestone by
    # index (original 56%-baseline behaviour).
    _focus_mode = os.environ.get("NIC_FOCUS_MODE", "index").lower()
    advice_branches: List[BranchRollout] = []
    for k in range(K):
        fresh_seed = step.crn_seed + 101 * (k + 1)
        if not milestones or _focus_mode in ("none", "plain", "off"):
            # ablation (f): plain resampling from pi^a(.|s_t), no milestone
            # conditioning. NIC_FOCUS_MODE=none disables the focus prefix.
            focus = None
        elif _focus_mode == "horizon":
            focus = milestones[:k + 1]          # ordered plan over next k+1 subgoals
        else:
            focus = milestones[k % len(milestones)]
        fell_back = False
        try:
            fresh_gate, fresh_advice, fresh_raw = companion.act(
                obs=step.state_dict,
                history=step.history,
                seed=fresh_seed,
                temperature=fresh_temperature,
                top_p=fresh_top_p,
                focus=focus,
            )
        except Exception as e:
            if debug:
                print(f"    [debug k={k}] companion.act EXCEPTION: {e}")
            fresh_advice = ""
            fresh_raw = ""
        # If the fresh sample happens to be SILENCE, force HELP and use its advice
        # if any; otherwise fall back to the main advice (rare — only if companion
        # deterministically outputs SILENCE for this state).
        if not fresh_advice:
            fresh_advice = step.advice_actual
            fresh_raw = step.companion_response_text
            fell_back = True
        if debug:
            marker = " (FELL BACK to main advice)" if fell_back else ""
            preview = fresh_advice.replace("\n", " ")[:80]
            print(f"    [debug k={k}] seed={fresh_seed} T={fresh_temperature} "
                  f"→ gate={('SILENCE' if fell_back else fresh_gate)}{marker}")
            print(f"      advice: '{preview}'")

        branch = rollout_branch(
            env=env, advisee=advisee, companion=companion,
            snapshot=step.snapshot,
            crn_seed=step.crn_seed,
            forced_gate="HELP",
            forced_advice=fresh_advice,
            max_steps=max_steps,
            branch_type="advice",
            initial_obs=initial_obs,
            initial_history=initial_history,
            advice_response_text=fresh_raw,
        )
        advice_branches.append(branch)

    return ExpandedHelpState(
        main_trajectory_id=(main_trajectory.worker_id, main_trajectory.episode_idx),
        step_index=step_index,
        snapshot_used=step.snapshot,
        crn_seed=step.crn_seed,
        silence_branch=silence_branch,
        help_replay_branch=help_replay_branch,
        advice_branches=advice_branches,
        main_advice_text=step.advice_actual,
    )


# ---------------------------------------------------------------------------
# Full-trajectory expansion
# ---------------------------------------------------------------------------

def expand_trajectory(
    env: ALFWorldEnv,
    advisee: VllmAdvisee,
    companion: VllmCompanionRaw,
    main_trajectory: MainTrajectory,
    K: int,
    max_steps: int,
    max_help_states: Optional[int] = None,
    subsample_mode: str = "uniform",     # "uniform" | "first"
    verbose: bool = False,
) -> List[ExpandedHelpState]:
    """Expand HELP states in a main trajectory.

    Args:
      max_help_states: cap on how many HELP states to expand per traj.
        None = expand all. Very important — a 40-step ep can have 30 HELP
        states, and expanding all takes 30×(K+2)×max_steps vLLM calls
        (easily 60+ minutes per traj at K=2, max_steps=20).
      subsample_mode:
        - "uniform": evenly-spaced picks across the traj (recommended,
          preserves state diversity)
        - "first": pick first N HELP states only (fast but biased to early
          states which are all similar)

    Returns list of ExpandedHelpState.
    """
    all_help = list(main_trajectory.help_step_indices)
    if max_help_states is not None and len(all_help) > max_help_states:
        if subsample_mode == "uniform":
            # Evenly-spaced picks
            indices = [int(round(i * (len(all_help) - 1) / (max_help_states - 1)))
                       for i in range(max_help_states)] if max_help_states > 1 else [0]
            picked = [all_help[i] for i in indices]
        else:
            picked = all_help[:max_help_states]
        if verbose:
            print(f"  [expand] traj=({main_trajectory.worker_id},"
                  f"{main_trajectory.episode_idx}) has {len(all_help)} HELP "
                  f"states — subsampled to {len(picked)} ({subsample_mode})",
                  flush=True)
    else:
        picked = all_help

    expanded: List[ExpandedHelpState] = []
    for i, step_idx in enumerate(picked):
        if verbose:
            print(f"  [expand] traj=({main_trajectory.worker_id},{main_trajectory.episode_idx}) "
                  f"[{i+1}/{len(picked)}] step={step_idx}", flush=True)
        result = expand_help_state(
            env=env, advisee=advisee, companion=companion,
            main_trajectory=main_trajectory,
            step_index=step_idx,
            K=K,
            max_steps=max_steps,
        )
        if result is not None:
            expanded.append(result)
    return expanded


# ---------------------------------------------------------------------------
# Sanity self-test (mock env)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Very light smoke — just verify import chain works."""
    print("[test] cf_expander imports OK")
    print("[test] structures:")
    print(f"       BranchRollout fields: {BranchRollout.__dataclass_fields__.keys()}")
    print(f"       ExpandedHelpState fields: {ExpandedHelpState.__dataclass_fields__.keys()}")


if __name__ == "__main__":
    _self_test()
