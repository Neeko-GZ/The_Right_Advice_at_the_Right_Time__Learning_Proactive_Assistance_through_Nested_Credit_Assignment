"""
train_cf_grpo.py — main entry for SPARK-Companion CF-nstep-GRPO training.

Two modes:

  --mode smoke     Load a saved List[ExpandedHelpState] pickle (from
                   test_cf_expander_smoke.py --save-expanded-pkl) and run
                   1 update step. Fast end-to-end check for content +
                   gate trainer wiring. Recommended for morning smoke test.

  --mode train     Full loop: for each update_step, run rollout →
                   expand → advantages → content_trainer + gate_trainer.
                   Serves rollout via existing vLLM endpoint (rollout
                   companion is FROZEN at the last saved checkpoint;
                   user manually restarts vLLM to hot-reload).

Reference: method.md § 4.2, § 5.5 + md/SPARK_finetune_unified.md v4.

Usage (smoke mode — the morning first test):
    PYTHONPATH=. python3 rl_causal/scripts/train_cf_grpo.py \\
        --mode smoke \\
        --expanded-pkl /tmp/expanded_smoke.pkl \\
        --main-traj-pkl /tmp/main_traj_smoke.pkl \\
        --base-model /workspace/models/Qwen3.5-4B \\
        --adapter-path /workspace/checkpoints/companion_sft_v3_C \\
        --value-head-path /workspace/checkpoints/critic_warmup/value_head.pt \\
        --output-dir /workspace/checkpoints/cf_grpo_smoke \\
        --n-updates 1 \\
        --K 2 \\
        --gamma 0.95

Usage (full train mode — post smoke):
    PYTHONPATH=. python3 rl_causal/scripts/train_cf_grpo.py \\
        --mode train \\
        --base-model /workspace/models/Qwen3.5-4B \\
        --adapter-path /workspace/checkpoints/companion_sft_v3_C \\
        --value-head-path /workspace/checkpoints/critic_warmup/value_head.pt \\
        --companion-url http://localhost:8001 --companion-model companion \\
        --advisee-url http://localhost:8000 --advisee-model /workspace/models/Qwen3.5-4B \\
        --output-dir /workspace/checkpoints/cf_grpo_v1 \\
        --n-updates 100 --n-episodes-per-update 8 --parallel 2 \\
        --K 3 --max-steps 20 --gamma 0.95 \\
        --alpha-intr 0.1 --alpha-div 0.05
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Ensure all classes needed for pickle.load are importable in this scope
# ---------------------------------------------------------------------------
from rl_causal.ppo.cf_rollout_worker import (
    MainStep, MainTrajectory, VllmCompanionRaw, rollout_worker,
)
from rl_causal.ppo.cf_expander import (
    ExpandedHelpState, BranchRollout, expand_trajectory,
)
from rl_causal.ppo.cf_token_masks import build_token_mask
from rl_causal.ppo.cf_nstep_q import (
    build_branch_from_rollout_data, compute_nstep_q, compute_q_targets_along_trajectory,
    BranchTrajectory,
)
from rl_causal.ppo.cf_reward_assembler import compute_group_shaping_rewards
from rl_causal.ppo.cf_advantage import batch_group_advantages
from rl_causal.ppo.cf_content_trainer import (
    ContentTrainer, ContentSample, build_content_samples,
)
from rl_causal.ppo.cf_gate_trainer import (
    GateTrainer, GateSample, build_gate_samples,
)


# ---------------------------------------------------------------------------
# vLLM dynamic LoRA hot-reload (v0.19+ endpoint style)
# ---------------------------------------------------------------------------

def remap_lora_for_vllm(src_dir: str) -> str:
    """Remap LoRA keys so vLLM (loaded with --language-model-only on the
    Qwen3.5-VL base) actually applies the adapter.

    Training saves keys under the text CausalLM view
    (base_model.model.model.layers.X...), but vLLM loads the model as
    Qwen3_5ForConditionalGeneration where the language layers live at
    base_model.model.model.language_model.layers.X... . Without this rename,
    vLLM silently drops the LoRA and serves the base model (the bug that made
    every rollout/eval use base weights). Renames only the keys; weights and
    adapter_config are unchanged. Returns a sibling '<src>_vllm' dir.
    """
    import os as _os
    import shutil as _sh
    import safetensors.torch as _st
    src = str(src_dir).rstrip("/")
    dst = src + "_vllm"
    _os.makedirs(dst, exist_ok=True)
    OLD = "base_model.model.model.layers."
    NEW = "base_model.model.model.language_model.layers."
    sd = _st.load_file(_os.path.join(src, "adapter_model.safetensors"))
    new = {(k.replace(OLD, NEW) if OLD in k else k): v for k, v in sd.items()}
    _st.save_file(new, _os.path.join(dst, "adapter_model.safetensors"))
    _sh.copy(_os.path.join(src, "adapter_config.json"),
             _os.path.join(dst, "adapter_config.json"))
    return dst


def hot_reload_companion_lora(
    companion_url: str,
    adapter_name: str,
    new_adapter_path: str,
    timeout: float = 30.0,
) -> bool:
    """Swap the vLLM adapter named `adapter_name` to point at `new_adapter_path`.

    vLLM v0.19+ endpoints (requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=True at
    server startup):
      DELETE /adapters/{name}        — unload
      POST   /adapters   body {name, src}  — load

    Sequence:
      1. DELETE old (ignore 404 if not registered)
      2. POST new (must return 200)

    Args:
      companion_url    — e.g. http://localhost:8001
      adapter_name     — the model name rollout uses (e.g. "companion")
      new_adapter_path — filesystem path to new checkpoint (must contain
                         adapter_config.json + adapter_model.safetensors)

    Returns:
      True on success, False on any failure (log printed to stdout).
    """
    base = companion_url.rstrip("/")

    # 1) Unload old. Use /v1/unload_lora_adapter — the /adapters endpoint only
    # registers metadata at runtime and does NOT actually apply the LoRA
    # (rollout silently falls back to BASE). /v1/*_lora_adapter is the path
    # that truly applies at runtime (verified via probe).
    try:
        r = requests.post(
            f"{base}/v1/unload_lora_adapter",
            json={"lora_name": adapter_name}, timeout=timeout,
        )
        if r.status_code == 200:
            print(f"  [hot-reload] unloaded old '{adapter_name}'", flush=True)
        else:
            # Not loaded yet / already gone — fine.
            pass
    except Exception as e:
        print(f"  [hot-reload] unload request failed: {e}", flush=True)
        return False

    # 2) Load new
    try:
        r = requests.post(
            f"{base}/v1/load_lora_adapter",
            json={"lora_name": adapter_name, "lora_path": new_adapter_path},
            timeout=timeout,
        )
        if r.status_code == 200:
            print(f"  [hot-reload] loaded '{adapter_name}' ← {new_adapter_path}",
                  flush=True)
            return True
        else:
            print(f"  [hot-reload] LOAD FAILED ({r.status_code}): {r.text[:200]}",
                  flush=True)
            return False
    except Exception as e:
        print(f"  [hot-reload] load request failed: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Critic forward on a raw prompt (needed for n-step Q tail bootstrap)
# ---------------------------------------------------------------------------

def critic_forward_batch(
    model,                       # CompanionWithValueHead
    tokenizer,
    prompts: List[str],
    max_length: int = 2048,
    mini_batch_size: int = 4,
    device=None,
) -> List[float]:
    """Batched V(s) forward on state-only prompts. Returns list of scalars."""
    import torch
    if device is None:
        device = next(model.parameters()).device
    out: List[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prompts), mini_batch_size):
            mb = prompts[start:start + mini_batch_size]
            # Tokenize + pad
            tok_lists = [
                tokenizer(p, add_special_tokens=False, truncation=True,
                           max_length=max_length)["input_ids"]
                for p in mb
            ]
            max_len = max(len(t) for t in tok_lists)
            pad_id = tokenizer.pad_token_id or 0
            input_ids = torch.tensor(
                [t + [pad_id] * (max_len - len(t)) for t in tok_lists],
                dtype=torch.long, device=device,
            )
            attn = torch.tensor(
                [[1] * len(t) + [0] * (max_len - len(t)) for t in tok_lists],
                dtype=torch.long, device=device,
            )
            _, values = model(input_ids=input_ids, attention_mask=attn, return_logits=False)
            out.extend([float(v) for v in values.detach().cpu().tolist()])
    return out


# ---------------------------------------------------------------------------
# Compute n-step Q targets along a MainTrajectory (for value-head training)
# ---------------------------------------------------------------------------

def apply_help_cost_to_rewards(
    rewards: List[float],
    gates: List[str],
    dt0: float,
    base: float,
    extra: float,
    tau: float,
    milestone_fired: Optional[List[bool]] = None,
) -> List[float]:
    """Subtract a recency-decayed intervention cost from each step whose gate
    was HELP, IN THE REWARD (so it flows through n-step Q for both gate and
    content, and every HELP in the rollout pays — not just the branch point):

        c_t = base + extra · exp(−Δt / tau)

    Δt = steps since the previous HELP. `dt0` seeds it (steps since the last
    HELP BEFORE this rollout began; math.inf if none → first HELP pays base).

    milestone_fired: if given, a completed subtask at step j RESETS Δt to ∞
    (cost → base for the next help). Rationale: after reaching a new subtask
    the situation has advanced, so an immediate next help is not redundant
    "consecutive spam" and its recency penalty is forgiven.
    """
    if (base == 0.0 and extra == 0.0) or not rewards:
        return list(rewards)
    out = list(rewards)
    since = dt0
    for j in range(len(out)):
        if j < len(gates) and gates[j] == "HELP":
            e = math.exp(-since / tau) if since != math.inf else 0.0
            out[j] -= (base + extra * e)
            since = 0.0
        since += 1.0
        if milestone_fired is not None and j < len(milestone_fired) and milestone_fired[j]:
            since = math.inf   # subtask reached → forgive recency, next help cheap
    return out


def compute_main_traj_q_targets(
    model,
    tokenizer,
    trajectory: MainTrajectory,
    gamma: float = 0.95,
    help_cost: float = 0.0,
    help_cost_extra: float = 0.0,
    help_cost_tau: float = 5.0,
) -> List[float]:
    """Compute n-step Q target at every step of a MainTrajectory.

    Formula (backward recursion, cf_nstep_q):
        Q(s_T) = V(s_T)  (tail bootstrap; V=0 if trajectory reached done)
        Q(s_k) = r_k + γ · Q(s_{k+1})  for k < T

    We use MC returns from executed rewards + critic bootstrap on the LAST
    state's prompt. This gives ONE Monte Carlo sample of V^π(s_k) at each
    step, which the value head then MSE-fits against. Matches how critic
    warmup was trained.

    Args:
      trajectory: a MainTrajectory
      gamma: discount factor

    Returns:
      List[float] of length trajectory.trajectory_len+1. Element i is
      the Q target for the state at step i.
    """
    raw_rewards = [step.r_env for step in trajectory.steps]
    if len(raw_rewards) == 0:
        return [0.0]
    # Intervention cost lives in the reward: subtract it at each HELP step of
    # the main trajectory (dt0=inf → the first HELP pays only the base).
    gates = [step.gate_actual for step in trajectory.steps]
    ms_fired = [bool(getattr(step, "milestone_fired", False)) for step in trajectory.steps]
    rewards = apply_help_cost_to_rewards(
        raw_rewards, gates, math.inf, help_cost, help_cost_extra, help_cost_tau,
        milestone_fired=ms_fired,
    )

    # Determine tail V(s_last):
    #   - If traj reached success (done), V(terminal) = 0
    #   - Else, forward critic on last step's companion prompt as a proxy
    #     for V at the post-last-action state. This isn't exact (we don't
    #     store the after-obs of the terminal step), but it's the closest
    #     approximation with data we have.
    if trajectory.success:
        v_tail = 0.0
        reached_done = True
    else:
        last_prompt = trajectory.steps[-1].companion_prompt
        v_tail = critic_forward_batch(model, tokenizer, [last_prompt])[0]
        reached_done = False

    bt = build_branch_from_rollout_data(
        rewards=rewards,
        state_values=[0.0] * len(rewards),   # unused by backward recursion
        reached_done=reached_done,
        critic_bootstrap_at_end=(None if reached_done else v_tail),
    )
    return compute_q_targets_along_trajectory(bt, gamma=gamma)


# ---------------------------------------------------------------------------
# Compute Q + shaping for a list of ExpandedHelpState
# ---------------------------------------------------------------------------

def compute_qs_and_shaping(
    model,
    tokenizer,
    expanded_states: List[ExpandedHelpState],
    state_dicts_by_state: Optional[Dict[int, Dict[str, Any]]] = None,
    gamma: float = 0.95,
    help_cost: float = 0.0,
    help_cost_extra: float = 0.0,
    help_cost_tau: float = 5.0,
    steps_since_help: Optional[Dict[int, float]] = None,
    verbose: bool = False,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """For each ExpandedHelpState, return (Qs, r_intr, r_div) per branch.

    Q_i:
      - If branch reached done: full MC over rewards (tail V=0).
      - If truncated: sum γ^j r_j + γ^H V(s_last), critic forward on
        branch.last_obs_text as a fallback (branch.state_prompts[-1] if
        available). We batch across all tail-bootstrap needs.

    Args:
      state_dicts_by_state: optional map {expanded_state_id: obs_dict}
        used for grounding_score. Without it, grounding falls back to 0.2.
        The obs_dict should have "text" and "admissible_commands" keys
        (same shape as MainStep.state_dict).
    """
    # Collect tail-bootstrap prompts across all branches that need it
    tail_prompts: List[str] = []
    tail_owners: List[Tuple[int, int]] = []  # (state_idx, branch_idx_flat)
    all_branches: List[List[BranchRollout]] = []
    for sidx, st in enumerate(expanded_states):
        branches = [st.silence_branch, st.help_replay_branch] + list(st.advice_branches)
        all_branches.append(branches)
        for bidx, br in enumerate(branches):
            if not br.reached_done and br.length > 0:
                # Prefer state_prompts[-1] if available (matches the exact
                # prompt shape critic warmup used); fall back to last_obs_text.
                if br.state_prompts:
                    prompt = br.state_prompts[-1]
                else:
                    prompt = br.last_obs_text
                tail_prompts.append(prompt)
                tail_owners.append((sidx, bidx))

    # Batched critic forward on all tails at once
    if tail_prompts:
        if verbose:
            print(f"  [critic] tail bootstrap on {len(tail_prompts)} branches",
                  flush=True)
        tail_values = critic_forward_batch(model, tokenizer, tail_prompts)
    else:
        tail_values = []
    tail_map: Dict[Tuple[int, int], float] = {
        tail_owners[i]: tail_values[i] for i in range(len(tail_prompts))
    }

    # Compute Qs
    per_state_qs: List[List[float]] = []
    per_state_intrs: List[List[float]] = []
    per_state_divs: List[List[float]] = []
    for sidx, st in enumerate(expanded_states):
        branches = all_branches[sidx]
        qs: List[float] = []
        # Intervention cost lives in the REWARD now (not a post-hoc Q subtraction):
        # each branch's HELP steps are charged a recency-decayed cost before Q,
        # so every HELP pays and it flows into both gate and content advantages.
        # dt0 = steps since the last HELP BEFORE this HELP state (seeds Δt).
        dt0 = steps_since_help.get(sidx, math.inf) if steps_since_help else math.inf
        for bidx, br in enumerate(branches):
            adj_rewards = apply_help_cost_to_rewards(
                br.rewards, br.gates, dt0,
                help_cost, help_cost_extra, help_cost_tau,
                milestone_fired=getattr(br, "milestone_fired", None),
            )
            if br.length == 0:
                # Degenerate — no rewards; treat Q as 0 (or tail if we had one)
                q = 0.0
            elif br.reached_done:
                # Full MC, tail = 0
                bt = build_branch_from_rollout_data(
                    rewards=adj_rewards,
                    state_values=[0.0] * len(adj_rewards),
                    reached_done=True,
                )
                q = compute_nstep_q(bt, gamma=gamma)
            else:
                # Truncated: tail bootstrap
                v_tail = tail_map.get((sidx, bidx), 0.0)
                bt = build_branch_from_rollout_data(
                    rewards=adj_rewards,
                    state_values=[0.0] * len(adj_rewards),
                    reached_done=False,
                    critic_bootstrap_at_end=float(v_tail),
                )
                q = compute_nstep_q(bt, gamma=gamma)

            qs.append(q)
        per_state_qs.append(qs)

        # Shaping rewards for the K+1 group at this state.
        # Note: SILENCE branch is at index 0, HELP-replay at 1, advice at 2..
        # Our advantage code expects silence_index=0 too.
        advice_texts = [br.advice_text for br in branches]
        raw_responses = [br.advice_response_text for br in branches]
        # Grounding needs obs+admissible from the HELP state's main-traj step.
        # If caller supplied state_dicts_by_state, use it; else fall back to None
        # (grounding_score returns 0.2 for non-empty advice with no state ref).
        state_dict = None
        if state_dicts_by_state is not None:
            state_dict = state_dicts_by_state.get(sidx)
        shaping = compute_group_shaping_rewards(
            advice_texts=advice_texts,
            raw_responses=raw_responses,
            state_dict=state_dict,
        )
        per_state_intrs.append([r.r_intrinsic for r in shaping])
        per_state_divs.append([r.r_diversity for r in shaping])

    return per_state_qs, per_state_intrs, per_state_divs


# ---------------------------------------------------------------------------
# One update step — pure function of (expanded_states, prompts, trainers)
# ---------------------------------------------------------------------------

def run_one_update_step(
    model,
    tokenizer,
    main_trajectories: List[MainTrajectory],
    expanded_states: List[ExpandedHelpState],
    content_trainer: ContentTrainer,
    gate_trainer: GateTrainer,
    gamma: float = 0.95,
    alpha_intr: float = 0.1,
    alpha_div: float = 0.05,
    help_cost: float = 0.0,
    help_cost_extra: float = 0.0,
    help_cost_tau: float = 5.0,
    train_mode: str = "joint",     # "content_only" | "gate_only" | "joint"
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute Q + advantages + train content and gate. Return metrics dict.

    Builds three cross-reference lookups from main_trajectories:
      * prompt_texts_by_state[sid]   → companion_prompt at that HELP step
      * state_dicts_by_state[sid]    → obs dict (for grounding_score)
      * q_targets_by_state[sid]      → (state_prompt, n-step Q target)
                                       used by content_trainer's value head loss
    """
    t0 = time.time()

    # 0. Build cross-reference from expanded HELP states to their main-traj step
    traj_by_id: Dict[Tuple[int, int], MainTrajectory] = {
        (t.worker_id, t.episode_idx): t for t in main_trajectories
    }
    prompt_texts_by_state: Dict[int, str] = {}
    state_dicts_by_state: Dict[int, Dict[str, Any]] = {}
    q_targets_by_state: Dict[int, Tuple[str, float]] = {}
    steps_since_help_by_state: Dict[int, float] = {}

    # 0a. Per-trajectory n-step Q targets (cache so we don't recompute per state)
    main_traj_q_targets: Dict[Tuple[int, int], List[float]] = {}
    for tid, tj in traj_by_id.items():
        main_traj_q_targets[tid] = compute_main_traj_q_targets(
            model, tokenizer, tj, gamma=gamma,
            help_cost=help_cost, help_cost_extra=help_cost_extra,
            help_cost_tau=help_cost_tau,
        )
    if verbose:
        n_trajs = len(main_traj_q_targets)
        print(f"  [q_targets] main-traj Q computed for {n_trajs} trajs "
              f"(tail bootstrap on unsuccessful ones)", flush=True)

    # 0b. Populate per-state lookups
    for sid, st in enumerate(expanded_states):
        tj = traj_by_id.get(st.main_trajectory_id)
        if tj is None or not (0 <= st.step_index < len(tj.steps)):
            continue
        step = tj.steps[st.step_index]
        prompt_texts_by_state[sid] = step.companion_prompt
        state_dicts_by_state[sid] = step.state_dict
        q_at_step = main_traj_q_targets[st.main_trajectory_id][st.step_index]
        q_targets_by_state[sid] = (step.companion_prompt, float(q_at_step))
        # Δt = steps since the previous HELP in this trajectory (∞ if first HELP).
        prev_helps = [h for h in tj.help_step_indices if h < st.step_index]
        steps_since_help_by_state[sid] = (
            float(st.step_index - max(prev_helps)) if prev_helps else math.inf
        )

    # 1. Q values + shaping rewards
    if verbose:
        print(f"[step] compute Q + shaping on {len(expanded_states)} states...",
              flush=True)
    per_state_qs, per_state_intrs, per_state_divs = compute_qs_and_shaping(
        model, tokenizer, expanded_states,
        state_dicts_by_state=state_dicts_by_state,
        gamma=gamma, help_cost=help_cost,
        help_cost_extra=help_cost_extra, help_cost_tau=help_cost_tau,
        steps_since_help=steps_since_help_by_state,
        verbose=verbose,
    )

    # 2. Advantages (K+1 group-relative + shaping)
    adv_batch = batch_group_advantages(
        list(zip(per_state_qs, per_state_intrs, per_state_divs)),
        alpha_intr=alpha_intr, alpha_div=alpha_div,
    )
    if verbose:
        gate_mean = sum(adv_batch.gate_advantages) / max(1, len(adv_batch.gate_advantages))
        q_target_vals = [q for _, q in q_targets_by_state.values()]
        if q_target_vals:
            q_stats = (f"mean={sum(q_target_vals)/len(q_target_vals):+.3f} "
                       f"min={min(q_target_vals):+.3f} "
                       f"max={max(q_target_vals):+.3f} "
                       f"n={len(q_target_vals)}")
        else:
            q_stats = "(none)"
        r_gnd = [ri for st_r in per_state_intrs for ri in st_r]
        r_gnd_mean = sum(r_gnd)/max(1,len(r_gnd))
        print(f"  [adv] {adv_batch.num_help_states} states, "
              f"{adv_batch.num_branch_samples} branch samples, "
              f"gate_adv_mean={gate_mean:+.3f}", flush=True)
        print(f"  [shaping] r_intr_mean={r_gnd_mean:.3f} "
              f"(grounding uses real state now)", flush=True)
        print(f"  [q_target_val] {q_stats}", flush=True)

    # 2b. ABLATION 3b (flat PPO): override the branch/group advantages with a
    # single actor-critic advantage A_t = Q_hat(s_t) - V_phi(s_t), applied
    # identically to the gate token and the replayed advice tokens (no branch
    # structure, no gate/content separation). Fresh branches (if any) get 0.
    if train_mode == "flat":
        sids = [s for s in range(len(expanded_states)) if s in q_targets_by_state]
        vprompts = [q_targets_by_state[s][0] for s in sids]
        vvals = critic_forward_batch(model, tokenizer, vprompts) if vprompts else []
        A_flat = {s: (q_targets_by_state[s][1] - float(vvals[i]))
                  for i, s in enumerate(sids)}
        adv_batch.gate_advantages = [
            A_flat.get(s, 0.0) for s in range(adv_batch.num_help_states)
        ]
        for i in range(len(adv_batch.flat_advantages)):
            sid = adv_batch.flat_state_ids[i]
            bid = adv_batch.flat_branch_ids[i]
            adv_batch.flat_advantages[i] = A_flat.get(sid, 0.0) if bid == 1 else 0.0
        if verbose:
            am = sum(adv_batch.gate_advantages) / max(1, len(adv_batch.gate_advantages))
            print(f"  [flat] A=Q_hat-V over {len(sids)} states, mean={am:+.3f}", flush=True)

    # 3. Build content samples + attach prompts + q_targets
    content_samples = build_content_samples(
        expanded_states, adv_batch,
        q_targets_by_state=q_targets_by_state,   # ← now populated
        skip_silence=(train_mode != "grpo"),     # grpo: keep all G free samples
    )
    for cs in content_samples:
        cs.prompt_text = prompt_texts_by_state.get(cs.state_id, "")

    # 4. Build gate samples
    gate_samples = build_gate_samples(expanded_states, adv_batch, prompt_texts_by_state)

    if verbose:
        print(f"  [samples] content={len(content_samples)}  gate={len(gate_samples)}",
              flush=True)

    # 5. Content trainer step (skipped if gate_only)
    t1 = time.time()
    # grpo: train the full {gate,advice} output jointly (gate not trained separately)
    content_trainer.full_output_mask = (train_mode == "grpo")
    if train_mode in ("joint", "content_only", "flat", "grpo"):
        if verbose:
            print(f"[step] content_trainer.step()...", flush=True)
        c_metrics = content_trainer.step(content_samples, verbose=verbose)
    else:
        c_metrics = {"n_samples": 0, "note": f"skipped (train_mode={train_mode})"}

    # 6. Gate trainer step (skipped if content_only)
    t2 = time.time()
    # grpo trains the gate jointly inside the content full-output mask, so no
    # separate gate pass here.
    if train_mode in ("joint", "gate_only", "flat"):
        if verbose:
            print(f"[step] gate_trainer.step()...", flush=True)
        g_metrics = gate_trainer.step(gate_samples, verbose=verbose)
    else:
        g_metrics = {"n_samples": 0, "note": f"skipped (train_mode={train_mode})"}

    dt = time.time() - t0
    return {
        "content": c_metrics,
        "gate": g_metrics,
        "n_help_states": adv_batch.num_help_states,
        "n_content_samples": len(content_samples),
        "n_gate_samples": len(gate_samples),
        "time_total_s": dt,
        "time_train_content_s": t2 - t1,
        "time_train_gate_s": time.time() - t2,
    }


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    base_model: str,
    adapter_path: Optional[str],
    value_head_path: Optional[str],
):
    """Load CompanionWithValueHead + tokenizer."""
    from rl_causal.critic import CompanionWithValueHead
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = CompanionWithValueHead.from_pretrained(
        base_model_path=base_model,
        adapter_path=adapter_path,
        value_head_path=value_head_path,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Smoke mode: load pickle, run 1 update step
# ---------------------------------------------------------------------------

def run_smoke_mode(args) -> int:
    print("=" * 78)
    print("train_cf_grpo — SMOKE mode (load pickle, 1 update step)")
    print("=" * 78)
    print(f"  expanded_pkl  : {args.expanded_pkl}")
    print(f"  main_traj_pkl : {args.main_traj_pkl}")
    print(f"  base_model    : {args.base_model}")
    print(f"  adapter       : {args.adapter_path}")
    print(f"  value_head    : {args.value_head_path}")
    print(f"  K             : {args.K}   gamma: {args.gamma}")
    print("=" * 78, flush=True)

    # Load pickled expanded states + main trajectories
    with open(args.expanded_pkl, "rb") as f:
        expanded: List[ExpandedHelpState] = pickle.load(f)
    with open(args.main_traj_pkl, "rb") as f:
        trajs: List[MainTrajectory] = pickle.load(f)
    print(f"  loaded {len(expanded)} expanded states from {len(trajs)} main trajs")

    # Load model
    print("[smoke] loading model + tokenizer (this may take 1-2 min)...", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.adapter_path, args.value_head_path,
    )

    # Trainers (shared optimizer via GateTrainer(optimizer=content.optimizer))
    content_trainer = ContentTrainer(
        model, tokenizer,
        lr_lora=args.lr_lora, lr_value=args.lr_value,
        ppo_clip=args.ppo_clip, value_coeff=args.value_coeff,
        ppo_epochs=args.ppo_epochs, mini_batch_size=args.mini_batch_size,
    )
    gate_trainer = GateTrainer(
        model, tokenizer,
        optimizer=content_trainer.optimizer,     # SHARED optimizer
        ppo_clip=args.ppo_clip, entropy_coeff=args.gate_entropy_coeff,
        ppo_epochs=args.ppo_epochs, mini_batch_size=args.mini_batch_size,
    )

    # Run 1 update
    for step_i in range(args.n_updates):
        print(f"\n{'='*78}\n[UPDATE {step_i+1}/{args.n_updates}]\n{'='*78}",
              flush=True)
        metrics = run_one_update_step(
            model, tokenizer,
            main_trajectories=trajs,
            expanded_states=expanded,
            content_trainer=content_trainer,
            gate_trainer=gate_trainer,
            gamma=args.gamma, alpha_intr=args.alpha_intr, alpha_div=args.alpha_div,
            help_cost=args.help_cost,
            help_cost_extra=args.help_cost_extra, help_cost_tau=args.help_cost_tau,
            train_mode=args.train_mode,
            verbose=True,
        )
        print(f"\n[metrics update {step_i+1}] (mode={args.train_mode})")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # Save checkpoint
    if args.output_dir:
        out = Path(args.output_dir) / "ckpt_smoke"
        out.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        print(f"[smoke] saved checkpoint → {out}")

    print("=" * 78)
    print("[smoke] done — if you see nonzero policy_loss and no crash, wiring works.")
    print("=" * 78)
    return 0


# ---------------------------------------------------------------------------
# Full train mode: rollout → expand → train, loop
# ---------------------------------------------------------------------------

def run_train_mode(args) -> int:
    """Full loop. Currently uses FROZEN vLLM companion for rollout — user
    must restart vLLM manually to hot-reload the newest checkpoint. Auto
    hot-reload is a TODO."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from rl_causal.advisee import VllmAdvisee
    from rl_causal.alfworld_env import ALFWorldEnv

    print("=" * 78)
    print("train_cf_grpo — FULL TRAIN mode")
    print("=" * 78, flush=True)

    # Load model
    print("[train] loading model + tokenizer...", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.adapter_path, args.value_head_path,
    )

    content_trainer = ContentTrainer(
        model, tokenizer,
        lr_lora=args.lr_lora, lr_value=args.lr_value,
        ppo_clip=args.ppo_clip, value_coeff=args.value_coeff,
        ppo_epochs=args.ppo_epochs, mini_batch_size=args.mini_batch_size,
    )
    gate_trainer = GateTrainer(
        model, tokenizer,
        optimizer=content_trainer.optimizer,
        ppo_clip=args.ppo_clip, entropy_coeff=args.gate_entropy_coeff,
        ppo_epochs=args.ppo_epochs, mini_batch_size=args.mini_batch_size,
    )

    # vLLM clients (rollout side is FROZEN — see docstring)
    companion_client = VllmCompanionRaw(
        model=args.companion_model, url=args.companion_url,
    )
    advisee_client = VllmAdvisee(
        model=args.advisee_model, url=args.advisee_url,
    )

    # Parse task-type exclusion set
    excluded_task_types = set(
        t.strip() for t in args.exclude_task_types.split(",") if t.strip()
    ) if args.exclude_task_types else set()
    if excluded_task_types:
        print(f"[train] excluding task types: {sorted(excluded_task_types)}", flush=True)

    # Persistent env for expansion — built ONCE, reused across updates.
    # Avoids re-scanning ALFWorld's 8810 game files on every update
    # (~11s per scan × N updates = wasted minutes).
    print("[train] building persistent expansion env (one-time, ~30s)...", flush=True)
    expansion_env = ALFWorldEnv(excluded_task_types=excluded_task_types)

    for update_step in range(args.n_updates):
        print(f"\n{'='*78}\n[UPDATE {update_step+1}/{args.n_updates}]\n{'='*78}",
              flush=True)
        t0 = time.time()

        # 1. Rollout main trajectories
        n_per_worker = max(1, args.n_episodes_per_update // args.parallel)
        workers_ep = [n_per_worker] * args.parallel
        for i in range(args.n_episodes_per_update - n_per_worker * args.parallel):
            workers_ep[i] += 1

        all_trajs: List[MainTrajectory] = []
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(
                    rollout_worker,
                    worker_id=w, n_episodes=workers_ep[w],
                    companion_url=args.companion_url, companion_model=args.companion_model,
                    advisee_url=args.advisee_url, advisee_model=args.advisee_model,
                    update_step=update_step,
                    max_steps=args.rollout_max_steps,
                    eps_gate_explore=args.eps_gate_explore,
                    seed_base=args.seed + w * 10000 + update_step * 100000,
                    excluded_task_types=excluded_task_types,
                ): w
                for w in range(args.parallel)
            }
            for fut in as_completed(futures):
                try:
                    all_trajs.extend(fut.result())
                except Exception as e:
                    print(f"  [rollout] worker failed: {e}", flush=True)
        # HELP rate diagnostic — key behavioral metric
        total_steps = sum(len(t.steps) for t in all_trajs)
        total_help = sum(len(t.help_step_indices) for t in all_trajs)
        n_success = sum(1 for t in all_trajs if t.success)
        help_rate = 100.0 * total_help / max(1, total_steps)
        succ_rate = 100.0 * n_success / max(1, len(all_trajs))
        print(f"  [rollout] {len(all_trajs)} trajs in {time.time()-t0:.0f}s  "
              f"| steps={total_steps} help={total_help} ({help_rate:.0f}%)  "
              f"success={n_success}/{len(all_trajs)} ({succ_rate:.0f}%)",
              flush=True)

        # 2. Expand HELP states (reuse persistent expansion_env)
        expanded: List[ExpandedHelpState] = []
        t_expand_start = time.time()
        print(f"  [expand] starting on {len(all_trajs)} trajs "
              f"({sum(len(t.help_step_indices) for t in all_trajs)} total HELP states, "
              f"K={args.K}, max_steps={args.branch_max_steps}) — may take several min...",
              flush=True)
        for traj_i, traj in enumerate(all_trajs):
            n_help = len(traj.help_step_indices)
            print(f"  [expand] traj {traj_i+1}/{len(all_trajs)}: "
                  f"{n_help} HELP states to expand...", flush=True)
            traj_expanded = expand_trajectory(
                expansion_env, advisee_client, companion_client, traj,
                K=args.K, max_steps=args.branch_max_steps,
                max_help_states=args.max_help_states_per_traj,
                subsample_mode=args.expand_subsample_mode,
                verbose=True,
            )
            expanded.extend(traj_expanded)
            print(f"  [expand] traj {traj_i+1} done ({time.time()-t_expand_start:.0f}s total)",
                  flush=True)
        print(f"  [expand] {len(expanded)} HELP states expanded", flush=True)

        if not expanded:
            print("  [warn] no HELP states this update — skipping training", flush=True)
            continue

        # 3-6. Q + advantages + train (all lookups built inside from all_trajs)
        metrics = run_one_update_step(
            model, tokenizer,
            main_trajectories=all_trajs,
            expanded_states=expanded,
            content_trainer=content_trainer,
            gate_trainer=gate_trainer,
            gamma=args.gamma, alpha_intr=args.alpha_intr, alpha_div=args.alpha_div,
            help_cost=args.help_cost,         # ← WAS MISSING (default 0.0 → no cost)
            help_cost_extra=args.help_cost_extra, help_cost_tau=args.help_cost_tau,
            train_mode=args.train_mode,       # ← WAS MISSING (default "joint" → gate trained anyway)
            verbose=(update_step < 3),   # verbose only for first 3 updates
        )
        dt_total = time.time() - t0
        print(f"\n[UPDATE {update_step+1}] total {dt_total:.0f}s")
        print(f"  content : loss={metrics['content'].get('policy_loss', 0):+.4f} "
              f"kl≈{metrics['content'].get('approx_kl', 0):+.4f}  "
              f"n={metrics['content'].get('n_samples', 0)}")
        print(f"  gate    : loss={metrics['gate'].get('gate_loss', 0):+.4f} "
              f"kl≈{metrics['gate'].get('approx_kl', 0):+.4f}  "
              f"n={metrics['gate'].get('n_samples', 0)}", flush=True)

        # Save every save_every updates + hot-reload vLLM companion
        if (update_step + 1) % args.save_every == 0:
            out = Path(args.output_dir) / f"ckpt_step_{update_step+1:04d}"
            out.mkdir(parents=True, exist_ok=True)
            model.save(str(out))
            print(f"  [save] → {out}", flush=True)

            # Hot-reload vLLM companion adapter so next update's rollout
            # uses the newly-trained policy (vs the frozen initial SFT-C).
            # Requires vLLM started with VLLM_ALLOW_RUNTIME_LORA_UPDATING=True.
            if args.hot_reload_companion:
                reload_path = str(out.absolute())
                if args.remap_lora_for_vllm:
                    try:
                        reload_path = remap_lora_for_vllm(reload_path)
                        print(f"  [remap] LoRA keys → {reload_path}", flush=True)
                    except Exception as e:
                        print(f"  [remap] FAILED ({e}) — loading un-remapped "
                              f"(vLLM will serve BASE, not the trained policy!)",
                              flush=True)
                ok = hot_reload_companion_lora(
                    companion_url=args.companion_url,
                    adapter_name=args.companion_model,
                    new_adapter_path=reload_path,
                )
                if not ok:
                    print(f"  [warn] hot-reload failed — subsequent rollouts "
                          f"will still use old policy in vLLM", flush=True)

    print("=" * 78)
    print("[train] done")
    print("=" * 78)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "train"], required=True)

    # Model
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--value-head-path", default=None)

    # Smoke-mode inputs
    ap.add_argument("--expanded-pkl", default=None)
    ap.add_argument("--main-traj-pkl", default=None)

    # Rollout (train mode)
    ap.add_argument("--exclude-task-types", default="pick_two_obj_and_place",
                    help="comma-separated ALFWorld task types to skip. Default "
                         "excludes pick_two_obj_and_place (very hard, sparse "
                         "reward, hurts learning). Set to '' to include all.")
    ap.add_argument("--companion-url", default="http://localhost:8001")
    ap.add_argument("--companion-model", default="companion")
    ap.add_argument("--advisee-url", default="http://localhost:8000")
    ap.add_argument("--advisee-model", default="/workspace/models/Qwen3.5-4B")
    ap.add_argument("--n-episodes-per-update", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--rollout-max-steps", type=int, default=20)
    ap.add_argument("--eps-gate-explore", type=float, default=0.10)

    # CF branching
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--branch-max-steps", type=int, default=8)
    ap.add_argument("--max-help-states-per-traj", type=int, default=6,
                    help="cap on HELP states to expand per traj (critical: "
                         "un-capped can blow up to 60+ min/update for long eps)")
    ap.add_argument("--expand-subsample-mode", choices=["uniform", "first"],
                    default="uniform",
                    help="how to pick HELP states when capped: 'uniform' "
                         "spreads across traj (preserves diversity), "
                         "'first' takes earliest")

    # Training
    ap.add_argument("--n-updates", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=0.95)
    # W33: both shaping terms OFF by default. Credit comes purely from the
    # counterfactual Q; r_intr (grounding is env-specific + leaky) and r_div
    # (lexical Jaccard; real diversity comes from milestone-anchored sampling)
    # are retained only as ablation knobs — pass a nonzero value to re-enable.
    ap.add_argument("--alpha-intr", type=float, default=0.0)
    ap.add_argument("--alpha-div", type=float, default=0.0)
    ap.add_argument("--help-cost", type=float, default=0.02,
                    help="BASE per-intervention cost subtracted from Q_i for "
                         "HELP branches (bidx>=1); biases A_gate toward SILENCE. "
                         "0=disable, 0.02=mild, 0.05=aggressive")
    ap.add_argument("--help-cost-extra", type=float, default=0.08,
                    help="EXTRA cost added right after a HELP, decaying with "
                         "steps-since-last-HELP: c_t = help_cost + "
                         "help_cost_extra*exp(-dt/tau). Penalizes CONSECUTIVE "
                         "help. 0=flat cost (old behavior).")
    ap.add_argument("--help-cost-tau", type=float, default=5.0,
                    help="decay constant (in steps) for --help-cost-extra. "
                         "Larger=longer cooldown before help is cheap again.")
    ap.add_argument("--lr-lora", type=float, default=1e-5)
    ap.add_argument("--lr-value", type=float, default=5e-5)
    ap.add_argument("--ppo-clip", type=float, default=0.2)
    ap.add_argument("--value-coeff", type=float, default=0.5)
    ap.add_argument("--gate-entropy-coeff", type=float, default=0.01)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--mini-batch-size", type=int, default=4)

    # Staged training
    ap.add_argument("--train-mode",
                    choices=["joint", "content_only", "gate_only", "flat", "grpo"],
                    default="joint",
                    help="content_only skips gate updates (Phase 1: let advice quality "
                         "improve first); gate_only skips content updates (Phase 2: "
                         "learn HELP/SILENCE with stable advice); joint trains both. "
                         "ABLATIONS: 'flat' = one n-step advantage A=Q_hat-V applied to "
                         "gate+advice alike, no branch/group structure (3b); 'grpo' = "
                         "iid group-normalized advantage over freely-sampled full "
                         "{gate,advice} outputs, no silence anchor / no gate-content "
                         "split (3c).")

    # Output
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hot-reload-companion", action="store_true", default=True,
                    help="After each save, DELETE + POST /adapters on vLLM 8001 "
                         "so rollout uses the newly-trained policy. Requires "
                         "vLLM started with VLLM_ALLOW_RUNTIME_LORA_UPDATING=True.")
    ap.add_argument("--no-hot-reload-companion", action="store_false",
                    dest="hot_reload_companion",
                    help="Disable hot-reload (rollout stays with initial vLLM adapter)")
    ap.add_argument("--remap-lora-for-vllm", action="store_true", default=True,
                    help="Before hot-reloading, remap LoRA keys to vLLM's "
                         "language_model.* naming (required for the Qwen3.5-VL "
                         "base + --language-model-only, else vLLM serves BASE).")
    ap.add_argument("--no-remap-lora-for-vllm", action="store_false",
                    dest="remap_lora_for_vllm",
                    help="Disable key remap (use only if the vLLM base is a true "
                         "text CausalLM whose module names already match).")

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "smoke":
        if not args.expanded_pkl or not args.main_traj_pkl:
            print("ERROR: --expanded-pkl and --main-traj-pkl required for smoke mode",
                  file=sys.stderr)
            return 1
        return run_smoke_mode(args)
    else:
        if not args.output_dir:
            print("ERROR: --output-dir required for train mode", file=sys.stderr)
            return 1
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        return run_train_mode(args)


if __name__ == "__main__":
    sys.exit(main())
