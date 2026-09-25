"""
train_mc.py — NIC (Nested Interventional Credit) training on Minecraft with a
VL companion (Option A / A2). SELF-CONTAINED MC entry; the ALFWorld
train_cf_grpo.py is left untouched.

Reuses the env-agnostic pieces:
  * rollout_one_episode  (cf_rollout_worker) — takes any env, here MCEnv
  * expand_trajectory    (cf_expander)       — snapshot/restore branches on MCEnv
  * compute_qs_and_shaping / batch_group_advantages — n-step Q + nested credit
New for VL:
  * CompanionWithValueHead(vision=True) + ContentTrainerVL / GateTrainerVL
  * VL sample builders that carry the HELP-state frames (state_dict["frames_b64"])

STATUS: written without a VL box to run on. This is the Stage-4 smoke entry;
run it small on Tokyo (--n-updates 1 --n-episodes 2) to surface Qwen3-VL
processor API issues, then scale.

Run:
    cd /workspace/credit && PYTHONPATH=. python3 rl_causal/scripts/train_mc.py \
      --base-model /workspace/models/Qwen3.5-9B \
      --adapter-path /workspace/checkpoints/cf_grpo_p2_v11/ckpt_step_0030 \
      --value-head-path /workspace/checkpoints/cf_grpo_p2_v11/ckpt_step_0030/value_head.pt \
      --spark-url http://localhost:8765 \
      --advisee-url http://localhost:8000 --advisee-model /workspace/models/Qwen3.5-4B \
      --companion-url http://localhost:8001 --companion-model /workspace/models/Qwen3.5-9B \
      --n-updates 1 --n-episodes 2 --K 2 --branch-max-steps 6
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Dict, List

from rl_causal.mc_env import MCEnv
from rl_causal.advisee import VllmAdvisee
from rl_causal.ppo.cf_rollout_worker import VllmCompanionRaw, rollout_one_episode
from rl_causal.ppo.cf_expander import expand_trajectory
from rl_causal.ppo.cf_advantage import batch_group_advantages
from rl_causal.critic import CompanionWithValueHead
from rl_causal.ppo.cf_content_trainer_vl import ContentTrainerVL, ContentSampleVL
from rl_causal.ppo.cf_gate_trainer_vl import GateTrainerVL, GateSampleVL
from rl_causal.scripts.train_cf_grpo import (
    compute_qs_and_shaping, compute_main_traj_q_targets,
    remap_lora_for_vllm, hot_reload_companion_lora,
)
from rl_causal.prompts.mc import SYSTEM_PROMPT_COMPANION, build_companion_prompt


def _help_state_context(traj, step_index):
    """Return (prompt_text, frames_b64, history) for the companion at a HELP state."""
    step = traj.steps[step_index]
    sd = dict(step.state_dict)
    frames = sd.get("frames_b64", []) or []
    sd_text = dict(sd)
    sd_text.pop("frames_b64", None)          # text-only prompt; frames passed separately
    prompt_text = build_companion_prompt(sd_text, history=step.history, history_window=10)
    if not isinstance(prompt_text, str):     # safety: should be str without frames
        prompt_text = str(prompt_text)
    return prompt_text, frames, step.history


def build_vl_samples(expanded_states, adv_batch, traj_by_id, q_targets_by_state):
    """Build ContentSampleVL (per non-silence branch) + GateSampleVL (per state)."""
    content: List[ContentSampleVL] = []
    gate: List[GateSampleVL] = []
    for sid, st in enumerate(expanded_states):
        traj = traj_by_id.get(st.main_trajectory_id)
        if traj is None or not (0 <= st.step_index < len(traj.steps)):
            continue
        prompt_text, frames, _hist = _help_state_context(traj, st.step_index)

        # gate sample (uses replay branch response)
        g_resp = st.help_replay_branch.advice_response_text or ""
        if g_resp:
            gate.append(GateSampleVL(
                prompt_text=prompt_text, response_text=g_resp,
                gate_advantage=float(adv_batch.gate_advantages[sid]),
                frames_b64=frames, system_prompt=SYSTEM_PROMPT_COMPANION, state_id=sid,
            ))
    # content samples from flat advantage arrays (skip silence bid==0)
    for flat_i, sid in enumerate(adv_batch.flat_state_ids):
        bid = adv_batch.flat_branch_ids[flat_i]
        adv = adv_batch.flat_advantages[flat_i]
        st = expanded_states[sid]
        traj = traj_by_id.get(st.main_trajectory_id)
        if traj is None or not (0 <= st.step_index < len(traj.steps)):
            continue
        if bid == 0:
            continue                          # silence: no advice tokens
        branch = st.help_replay_branch if bid == 1 else st.advice_branches[bid - 2]
        resp = branch.advice_response_text or ""
        if not resp:
            continue
        prompt_text, frames, _ = _help_state_context(traj, st.step_index)
        qt = None
        if bid == 1 and sid in q_targets_by_state:
            qt = q_targets_by_state[sid]
        content.append(ContentSampleVL(
            prompt_text=prompt_text, response_text=resp, advantage=float(adv),
            frames_b64=frames, system_prompt=SYSTEM_PROMPT_COMPANION,
            q_target=qt, state_id=sid, branch_id=bid, branch_type=branch.branch_type,
        ))
    return content, gate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="/workspace/models/Qwen3.5-9B")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--value-head-path", default=None)
    ap.add_argument("--create-lora", action="store_true",
                    help="MC init: base VL + a FRESH zero-init LoRA (recommended). "
                         "Ignores --adapter-path (v11 is ALFWorld-text, wrong for MC).")
    ap.add_argument("--spark-url", default="http://localhost:8765")
    ap.add_argument("--advisee-url", default="http://localhost:8000")
    ap.add_argument("--advisee-model", default="/workspace/models/Qwen3.5-4B")
    ap.add_argument("--companion-url", default="http://localhost:8001")
    ap.add_argument("--companion-model", default="/workspace/models/Qwen3.5-9B")
    ap.add_argument("--task", default=None,
                    help="single fixed task; if unset, sample per-episode from --tasks")
    ap.add_argument("--tasks",
                    default="collect_wood,craft_crafting_table,craft_wooden_pickaxe,"
                            "collect_stone,craft_stone_pickaxe",
                    help="comma-separated simple-task curriculum sampled per episode "
                         "(wood + stone tier). Ignored if --task is set.")
    ap.add_argument("--n-updates", type=int, default=1)
    ap.add_argument("--n-episodes", type=int, default=2)
    ap.add_argument("--rollout-max-steps", type=int, default=20)
    ap.add_argument("--branch-max-steps", type=int, default=6)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--max-help-states", type=int, default=4,
                    help="cap HELP states expanded per trajectory. MC branches run "
                         "REAL bot skills (snapshot+restore+K+2 branches x steps), so "
                         "this dominates wall-time — keep small (2-4 for smoke).")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--help-cost", type=float, default=0.005)
    ap.add_argument("--help-cost-extra", type=float, default=0.02)
    ap.add_argument("--help-cost-tau", type=float, default=5.0)
    ap.add_argument("--min-step-seconds", type=float, default=5.0,
                    help="pace each MAIN-rollout step to >= this many seconds "
                         "(branch expansion runs full-speed regardless). This is a "
                         "DESIGN choice: it sets a realistic decision cadence so the "
                         "companion's learned help-timing transfers to a real (e.g. "
                         "human) advisee. 0 = full speed (quick code smoke only).")
    ap.add_argument("--train-mode", choices=["joint", "content_only", "gate_only"],
                    default="joint",
                    help="staged training (matches the method): content_only lets "
                         "advice quality warm up first; gate_only learns when-to-help "
                         "with stable advice; joint trains both. Stage by running "
                         "content_only, then resuming with joint/gate_only from its ckpt.")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--eval-episodes", type=int, default=10,
                    help="after training, eval the final policy for this many episodes "
                         "in model-gate (ours) + silence (floor) modes. 0 = skip.")
    ap.add_argument("--hot-reload", action="store_true", default=True,
                    help="after each save, swap the vLLM companion adapter to the new "
                         "checkpoint so rollout tracks the trained policy (on-policy). "
                         "Requires vLLM 8001 with VLLM_ALLOW_RUNTIME_LORA_UPDATING=True, "
                         "and --companion-model must be the adapter NAME (e.g. 'companion').")
    ap.add_argument("--no-hot-reload", action="store_false", dest="hot_reload")
    ap.add_argument("--remap-lora", action="store_true", default=False,
                    help="remap LoRA keys for vLLM. A FRESH VL LoRA (--create-lora) is "
                         "built on the full VL model so its keys already match; leave OFF. "
                         "Only needed for a text (v11-style) LoRA.")
    args = ap.parse_args()

    print("=" * 74)
    print("train_mc — NIC on Minecraft (VL companion, A2)")
    print("=" * 74, flush=True)

    print("[mc] loading VL companion + value head...", flush=True)
    model = CompanionWithValueHead.from_pretrained(
        base_model_path=args.base_model,
        adapter_path=None if args.create_lora else args.adapter_path,
        value_head_path=None if args.create_lora else args.value_head_path,
        vision=True, create_lora=args.create_lora,
    )
    tok = model.processor.tokenizer

    content_trainer = ContentTrainerVL(model)
    gate_trainer = GateTrainerVL(model, optimizer=content_trainer.optimizer)

    from rl_spark.task_pool import get_task_by_name
    task_list = [args.task] if args.task else [t.strip() for t in args.tasks.split(",") if t.strip()]
    # Rollout-loop bound must cover the longest task's own budget, else the loop
    # cuts long tasks short even though the env would allow more steps.
    loop_max = max(int(args.rollout_max_steps),
                   max(int(getattr(get_task_by_name(t), "max_steps", 0) or 0) for t in task_list))
    print(f"[mc] task curriculum: {task_list}  (loop_max_steps={loop_max})", flush=True)
    env = MCEnv(base_url=args.spark_url, task_name=task_list[0],
                max_steps=loop_max,
                min_step_seconds=args.min_step_seconds)
    advisee = VllmAdvisee(model=args.advisee_model, url=args.advisee_url, env="minecraft")
    companion = VllmCompanionRaw(model=args.companion_model, url=args.companion_url,
                                 env="minecraft")

    def _reload(ckpt_dir):
        """Swap the vLLM companion adapter to ckpt_dir so rollout is on-policy."""
        if not args.hot_reload:
            return
        path = ckpt_dir
        if args.remap_lora:
            try:
                path = remap_lora_for_vllm(ckpt_dir)
            except Exception as e:
                print(f"  [remap] failed: {e}", flush=True)
        ok = hot_reload_companion_lora(args.companion_url, args.companion_model, path)
        print(f"  [hot-reload] {'ok' if ok else 'FAILED'} → {path}", flush=True)

    # Push the fresh (zero-init) LoRA into the vLLM companion slot so the very
    # first rollout matches the training model (both = base VL behavior at init).
    if args.hot_reload:
        init_dir = f"{args.output_dir or '/tmp/mc_vl'}/mc_ckpt_init"
        print(f"[mc] seeding vLLM companion slot with fresh LoRA → {init_dir}", flush=True)
        model.save(init_dir)
        _reload(init_dir)

    for upd in range(args.n_updates):
        print(f"\n{'='*74}\n[UPDATE {upd+1}/{args.n_updates}]\n{'='*74}", flush=True)
        t0 = time.time()
        # 1. rollout (paced for viewing if --min-step-seconds > 0)
        env.pacing_enabled = True
        trajs = []
        for ep in range(args.n_episodes):
            env.task_name = random.choice(task_list)   # sample the episode's task
            advisee.reset_history()
            traj = rollout_one_episode(
                env, companion, advisee, update_step=upd, worker_id=0, episode_idx=ep,
                max_steps=loop_max,
                seed=1000 + upd * 100 + ep, eps_gate_explore=0.10,
                exploration_rng=random.Random(1000 + upd * 100 + ep),
            )
            trajs.append(traj)
        n_help = sum(len(t.help_step_indices) for t in trajs)
        print(f"  [rollout] {len(trajs)} eps, {n_help} HELP states, {time.time()-t0:.0f}s",
              flush=True)

        # 2. expand HELP states (branches run full-speed regardless of pacing)
        env.pacing_enabled = False
        expanded = []
        for traj in trajs:
            expanded.extend(expand_trajectory(
                env, advisee, companion, traj,
                K=args.K, max_steps=args.branch_max_steps,
                max_help_states=args.max_help_states, verbose=True,
            ))
        if not expanded:
            print("  [warn] no HELP states expanded — skip update", flush=True)
            continue
        print(f"  [expand] {len(expanded)} HELP states", flush=True)

        # 3. cross-refs + n-step Q targets (critic tail is text-only for the smoke)
        traj_by_id = {(t.worker_id, t.episode_idx): t for t in trajs}
        state_dicts_by_state, q_targets_by_state, steps_since_help = {}, {}, {}
        main_q = {tid: compute_main_traj_q_targets(
                    model, tok, tj, gamma=args.gamma, help_cost=args.help_cost,
                    help_cost_extra=args.help_cost_extra, help_cost_tau=args.help_cost_tau)
                  for tid, tj in traj_by_id.items()}
        for sid, st in enumerate(expanded):
            tj = traj_by_id.get(st.main_trajectory_id)
            if tj is None or not (0 <= st.step_index < len(tj.steps)):
                continue
            state_dicts_by_state[sid] = tj.steps[st.step_index].state_dict
            q_targets_by_state[sid] = float(main_q[st.main_trajectory_id][st.step_index])
            prev = [h for h in tj.help_step_indices if h < st.step_index]
            steps_since_help[sid] = float(st.step_index - max(prev)) if prev else float("inf")

        # 4. Q per branch + nested advantages
        per_qs, per_intr, per_div = compute_qs_and_shaping(
            model, tok, expanded, state_dicts_by_state=state_dicts_by_state,
            gamma=args.gamma, help_cost=args.help_cost,
            help_cost_extra=args.help_cost_extra, help_cost_tau=args.help_cost_tau,
            steps_since_help=steps_since_help,
        )
        adv_batch = batch_group_advantages(list(zip(per_qs, per_intr, per_div)),
                                           alpha_intr=0.0, alpha_div=0.0)

        # 5. build VL samples + train
        content_samples, gate_samples = build_vl_samples(
            expanded, adv_batch, traj_by_id, q_targets_by_state)
        print(f"  [samples] content={len(content_samples)} gate={len(gate_samples)} "
              f"gate_adv_mean={sum(adv_batch.gate_advantages)/max(1,len(adv_batch.gate_advantages)):+.3f}",
              flush=True)

        if args.train_mode in ("joint", "content_only"):
            c_metrics = content_trainer.step(content_samples, verbose=True)
        else:
            c_metrics = {"skipped": args.train_mode}
        if args.train_mode in ("joint", "gate_only"):
            g_metrics = gate_trainer.step(gate_samples, verbose=True)
        else:
            g_metrics = {"skipped": args.train_mode}
        print(f"  [content] {c_metrics}")
        print(f"  [gate]    {g_metrics}", flush=True)

        if args.output_dir:
            out = f"{args.output_dir}/mc_ckpt_step_{upd+1:04d}"
            model.save(out)
            print(f"  [save] {out}", flush=True)
            _reload(out)   # next update's rollout uses the just-trained policy

    # ---- auto-eval the final policy (model-gate = ours, silence = floor) ----
    if args.eval_episodes > 0:
        from rl_causal.scripts.mc_eval import run_episode
        env.pacing_enabled = False    # eval full-speed
        for mode in ("model-gate", "silence"):
            comp = None if mode == "silence" else companion
            wins = help_tot = step_tot = 0
            ms_sum = 0.0
            n = args.eval_episodes
            print(f"\n[eval mode={mode}] {n} episodes...", flush=True)
            for i in range(n):
                task = task_list[i % len(task_list)]
                try:
                    r = run_episode(env, advisee, comp, mode, task, seed=9000 + i)
                except Exception as e:
                    print(f"  ep{i+1} eval error: {e}", flush=True); continue
                wins += int(r["won"]); help_tot += r["help"]; step_tot += r["steps"]
                ms_sum += r["ms_frac"]
                print(f"  ep{i+1} [{task}] {'WIN ' if r['won'] else 'fail'} "
                      f"help={r['help']} steps={r['steps']} ms={r['ms_frac']:.2f}", flush=True)
            print(f"[eval {mode}] SR={wins}/{n} ({100*wins/max(1,n):.0f}%)  "
                  f"HELP={help_tot}/{step_tot} ({100*help_tot/max(1,step_tot):.0f}%)  "
                  f"ms_frac={ms_sum/max(1,n):.2f}", flush=True)

    print("\n[train_mc] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
