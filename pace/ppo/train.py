"""
train.py — PPO online training entry point for the SPARK-track RL.

Wires together:
    rl_spark.policy.Policy       — Qwen3.5-VL + LoRA + SPARK projector + value_head
    rl_spark.spark_scorer.SparkScorer — reward source (Δscore from SPARK)
    rl_spark.mindcraft_env.MindcraftEnv — real Minecraft env via SparkAPI
    rl_spark.ppo.rollout         — collect_episode + flatten_episodes
    rl_spark.ppo.advantages      — GAE
    rl_spark.ppo.losses          — policy / value / kl loss
    rl_spark.ppo.value_head      — independent critic MLP

PPO loop per update:
    1. Snapshot a frozen ref policy (LoRA state at start of update) for KL.
    2. Collect N episodes by stepping MindcraftEnv with current policy.
       Each step:
         a) Policy.decide(features) → HELP/SILENCE
         b) if HELP: Policy.generate_advice(features) → advice text + response_ids
         c) MindcraftEnv.step(advice) → reward = SparkScorer Δscore + shaping
    3. Compute GAE per episode → advantages + returns.
    4. Flatten transitions across episodes into a FlatBatch.
    5. Run K PPO epochs over the FlatBatch, sample-by-sample:
         a) Re-forward policy (with grad) → logp_new
         b) Re-forward ref policy (no grad) → logp_ref
         c) Re-forward value head → v_new
         d) compute (policy_loss, value_loss, kl_loss)
         e) total.backward(); optim.step()
    6. Save per-epoch ckpt (LoRA + projector + value_head).

Notes:
  * We forward one sample at a time during the update step. SPARK + Qwen
    forward are already heavy and per-sample dataloading keeps the code
    simple (no padding-mask plumbing for variable-length advice responses).
  * Ref policy: we deep-copy ONLY the LoRA state dict at update start.
    The frozen base + projector + value_head are shared with the live
    policy by reference (saves memory; they're not what KL is about).
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from rl_spark.policy import Policy, PolicyConfig
from rl_spark.mindcraft_env import MindcraftEnv
from rl_spark.spark_scorer import SparkScorer
from rl_spark.reward import RewardConfig
from rl_spark.ppo.rollout import (
    Transition,
    EpisodeRollout,
    FlatBatch,
    collect_episode,
    flatten_episodes,
)
from rl_spark.ppo.advantages import compute_gae, standardize
from rl_spark.ppo.losses import (
    PPOLossConfig,
    policy_loss,
    value_loss,
    kl_against_ref,
    gather_response_logprobs,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class PPOTrainConfig:
    # -- env --
    spark_api_url: str = "http://localhost:8765"
    task: str = "collect wood"
    step_seconds: float = 4.0
    max_steps: int = 30
    n_frames: int = 16
    # how advice is delivered to the bot:
    #   "incoming" — simulates an external player saying it →
    #                bot.handleMessage triggers the LLM loop. THE RIGHT
    #                CHOICE for RL: bot actually processes + acts on advice.
    #   "chat"     — bot.chat() echoes the text; bot's self-chat filter
    #                ignores it. Diagnostic only.
    #   "history"  — silent inject into history; doesn't kick LLM cycle.
    advice_mode: str = "incoming"
    # if non-empty, on each reset() we POST this command via
    # /execute_command to put the bot into autonomous play mode (so the
    # advice we then send via /advice has something to steer).
    initial_goal_template: str = '!goal("Play Minecraft and {task}.")'

    # Per-episode reset behavior. Forwarded to SparkAPI /reset on each
    # env.reset(). Clearing inventory + history gives every episode a
    # clean starting state so:
    #   - task_progress reward computes against a 0-base each episode
    #   - bot LLM history doesn't accumulate across episodes (otherwise
    #     after a few episodes the prompt gets bloated with stale advice)
    #   - bot's behavior stays in the "fresh start" distribution that
    #     matches our task prompt ("collect wood")
    # restore_health + stop_actions ensure no carryover damage / actions.
    reset_kwargs: dict = field(default_factory=lambda: {
        "clear_inventory": True,
        "reset_history": True,
        "restore_health": True,
        "stop_actions": True,
    })

    # -- spark scorer (reward) --
    mineclip_ckpt: str = "/workspace/MineCLIP/attn.pth"
    spark_trainable_ckpt: str = "checkpoints/spark_pretrain_moga/spark_ep02.pt"
    score_mode: str = "ls_sigmoid"        # "ls_sigmoid" → [0,1]; "raw" → raw logit

    # -- policy --
    qwen_path: str = "/workspace/models/Qwen3.5-9B"
    lora_rank: int = 16
    lora_alpha: int = 32

    # -- rollout --
    # n_episodes_per_update bumped 2 → 4 to halve KL/ratio variance.
    # Smoke tests showed clipfrac stuck at 0.7-0.8 even with conservative
    # lr_lora=3e-6 because each minibatch had only 1-3 HELP samples —
    # gradient estimate was noisy. Doubling episodes → ~doubled HELP
    # samples per minibatch → tighter PPO updates. Update wall time goes
    # from ~150s to ~300s, acceptable.
    n_episodes_per_update: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    standardize_advantages: bool = True

    # -- PPO update --
    # Calibrated after smoke-test observed clipfrac=0.64, ratio_mean=1.68,
    # KL hitting 0.95 in the first epoch. Old defaults were too aggressive
    # for our small-batch regime (~10 HELP samples per update).
    n_epochs: int = 2          # 4 → 2: less reuse of same data per update
    minibatch_size: int = 4    # 16 → 4: take more update steps per epoch,
                                #         each with smaller gradient
    lr_lora: float = 1e-6      # 3e-6 → 1e-6: round-1 PPO showed
                                # clipfrac=0.619, kl=0.229 in epoch-0 minibatch-0
                                # (early-stop hit). Both ~3× too large
                                # for 4-episode batch → cut LR 3×.
    lr_projector: float = 3e-5 # 5e-5 → 3e-5: scale with lr_lora
    lr_value: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # -- loss coefficients (mostly from PPOLossConfig defaults) --
    clip_range: float = 0.1    # 0.2 → 0.1: tighter PPO clip = smaller steps
    clip_range_vf: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    kl_coef: float = 0.05
    target_kl: Optional[float] = 0.1  # 0.05 → 0.1: tolerate larger drift
                                       # (small batch → noisy KL estimate)

    # -- training loop --
    n_updates: int = 200
    save_every_updates: int = 5    # 10 → 5: save more often for safety
    save_dir: str = "checkpoints/ppo_spark"
    log_every_steps: int = 1
    device: str = "cuda:0"
    seed: int = 0

    # Resume from a previous PPO checkpoint (path to .pt saved by
    # policy.save_trainable). Loads LoRA + projector + value_head only;
    # SPARK + Qwen base + MineCLIP always come from their own paths.
    # Set to None (default) to start fresh. CLI: --resume-from <path>
    resume_from: Optional[str] = None


# ---------------------------------------------------------------------------
# PPO update step
# ---------------------------------------------------------------------------

def _update_step(
    policy: Policy,
    batch: FlatBatch,
    optimizer: torch.optim.Optimizer,
    loss_cfg: PPOLossConfig,
    cfg: PPOTrainConfig,
) -> dict:
    """One mini-batch PPO update over `batch`. Returns log dict (averaged).

    Features dicts (with PIL frames inside) live in `batch.features_list`
    and are forwarded one sample at a time — no padding-mask plumbing for
    variable-length advice responses.
    """
    device = torch.device(cfg.device)
    B = batch.size

    pol_losses, val_losses, kl_losses = [], [], []
    proj_gns = []
    ratios, clipfracs, approx_kls = [], [], []

    for i in range(B):
        # Skip SILENCE samples for policy/KL loss (no response → no ratio).
        # They still contribute to value loss.
        features_i = batch.features_list[i]
        is_help = bool(batch.is_help[i].item())

        # ---- value forward (always) ----
        v_new = policy.value(features_i).view(())
        v_old = batch.values_old[i]
        v_target = batch.returns[i]

        v_clipped = v_old + torch.clamp(v_new - v_old, -loss_cfg.clip_range_vf, loss_cfg.clip_range_vf)
        v_loss_i = 0.5 * torch.max(
            (v_new - v_target).pow(2),
            (v_clipped - v_target).pow(2),
        )

        # ---- policy + KL forward (only HELP samples have a response) ----
        if is_help and batch.response_ids[i].shape[0] > 0:
            response_ids_t = batch.response_ids[i].unsqueeze(0)
            logp_old_seq = batch.logp_olds[i].sum()
            advantage = batch.advantages[i]

            # logp under current policy (with grad)
            logits_new, prompt_length = policy.forward_logits_with_response(
                features_i, response_ids_t,
                prompt_kind="advice", task=features_i["task"],
            )
            logp_new = gather_response_logprobs(
                logits_new, response_ids_t, prompt_length,
            )
            seq_logp_new = logp_new.sum(dim=-1).squeeze(0)

            # logp under ref policy: ref_lora_state was snapshotted BEFORE
            # rollout, so the rollout policy == ref policy → logp_old_seq
            # is exactly logp_ref. Reusing it (a) avoids a second forward
            # and (b) avoids in-place LoRA swap that corrupted the autograd
            # graph (the swap would bump LoRA weight versions between this
            # forward and backward → "expected version 2, got 4").
            seq_logp_ref = logp_old_seq.detach()

            log_ratio = seq_logp_new - logp_old_seq
            ratio = log_ratio.exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio,
                1 - loss_cfg.clip_range, 1 + loss_cfg.clip_range,
            ) * advantage
            pol_loss_i = -torch.min(surr1, surr2)

            # KL k1 estimator (per-token, mean)
            kl_loss_i = (seq_logp_new - seq_logp_ref) / max(1, response_ids_t.shape[1])

            with torch.no_grad():
                ratios.append(float(ratio.item()))
                clipfracs.append(float(((ratio - 1.0).abs() > loss_cfg.clip_range).float().item()))
                approx_kls.append(float((-log_ratio).item()))
        else:
            # SILENCE: zero policy/KL contribution
            pol_loss_i = torch.zeros((), device=device)
            kl_loss_i = torch.zeros((), device=device)

        # ---- combine + backward ----
        total_i = (
            pol_loss_i
            + loss_cfg.vf_coef * v_loss_i
            + loss_cfg.kl_coef * kl_loss_i
        )
        optimizer.zero_grad(set_to_none=True)
        total_i.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy.trainable_parameters()),
            cfg.grad_clip,
        )
        # Diagnostic: projector grad_norm — if 0, SPARK pre-hook injection
        # is not connecting the projector graph to the loss (i.e., SPARK
        # placeholders never appear in input_ids, or hook didn't fire).
        proj_grads = [p.grad for p in policy.projector.parameters() if p.grad is not None]
        proj_gn = (
            float(torch.sqrt(sum(g.pow(2).sum() for g in proj_grads)))
            if proj_grads else 0.0
        )
        optimizer.step()

        pol_losses.append(float(pol_loss_i.detach().item()))
        val_losses.append(float(v_loss_i.detach().item()))
        kl_losses.append(float(kl_loss_i.detach().item()))
        proj_gns.append(proj_gn)

    out = {
        "loss/policy": float(np.mean(pol_losses)),
        "loss/value": float(np.mean(val_losses)),
        "loss/kl": float(np.mean(kl_losses)),
        "grad/projector": float(np.mean(proj_gns)),
    }
    if ratios:
        out["ppo/ratio_mean"] = float(np.mean(ratios))
        out["ppo/clipfrac"] = float(np.mean(clipfracs))
        out["ppo/approx_kl_old_new"] = float(np.mean(approx_kls))
    return out


# ---------------------------------------------------------------------------
# train loop
# ---------------------------------------------------------------------------

def train_loop(cfg: PPOTrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # --- env ---
    log.info("building MindcraftEnv + SparkScorer...")
    scorer = SparkScorer(
        task=cfg.task,
        mineclip_ckpt=cfg.mineclip_ckpt,
        trainable_ckpt=cfg.spark_trainable_ckpt,
        device=cfg.device,
        return_raw_logit=(cfg.score_mode == "raw"),
    )
    env = MindcraftEnv(
        scorer=scorer,
        base_url=cfg.spark_api_url,
        step_seconds=cfg.step_seconds,
        max_steps=cfg.max_steps,
        n_frames=cfg.n_frames,
        advice_mode=cfg.advice_mode,
        reset_kwargs=cfg.reset_kwargs,   # ← clear_inventory + history per episode
    )
    # Bot doesn't auto-start; kick it into autonomous play mode after each
    # reset by sending a goal command via /execute_command. With multi-task,
    # the goal command differs per episode — we wrap env.reset to read the
    # current goal from env._pending_goal_cmd (set by the rollout driver
    # in _run_main_loop before each episode).
    _original_reset = env.reset

    def _reset_with_dynamic_goal(**kwargs):
        obs, info = _original_reset(**kwargs)
        goal_cmd = getattr(env, "_pending_goal_cmd", None)
        if goal_cmd:
            try:
                env.client.execute_command(goal_cmd)
                log.info(f"sent initial goal command: {goal_cmd}")
            except Exception as e:
                log.warning(f"initial goal command failed: {e}")
        return obs, info

    env.reset = _reset_with_dynamic_goal

    # --- policy ---
    log.info("building Policy...")
    policy_cfg = PolicyConfig(
        qwen_path=cfg.qwen_path,
        spark_trainable_ckpt=cfg.spark_trainable_ckpt,
        mineclip_ckpt=cfg.mineclip_ckpt,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        device=cfg.device,
    )
    policy = Policy(policy_cfg)

    # Optionally resume LoRA + projector + value_head from a previous
    # PPO checkpoint. Other components (Qwen base, SPARK, MineCLIP)
    # always load from their own configured paths.
    if cfg.resume_from:
        log.info(f"resuming from PPO checkpoint: {cfg.resume_from}")
        policy.load_trainable(cfg.resume_from)

    # --- optimizer with per-group lr ---
    from peft import get_peft_model_state_dict
    lora_param_names = set(get_peft_model_state_dict(policy.qwen).keys())
    lora_params, proj_params, value_params = [], [], []
    for name, p in policy.qwen.named_parameters():
        if p.requires_grad:
            lora_params.append(p)
    for p in policy.projector.parameters():
        if p.requires_grad:
            proj_params.append(p)
    for p in policy.value_head.parameters():
        if p.requires_grad:
            value_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params,  "lr": cfg.lr_lora},
            {"params": proj_params,  "lr": cfg.lr_projector},
            {"params": value_params, "lr": cfg.lr_value},
        ],
        weight_decay=cfg.weight_decay,
    )
    log.info(
        f"optimizer: LoRA {len(lora_params)} tensors @ {cfg.lr_lora}, "
        f"projector {len(proj_params)} @ {cfg.lr_projector}, "
        f"value {len(value_params)} @ {cfg.lr_value}"
    )

    loss_cfg = PPOLossConfig(
        clip_range=cfg.clip_range,
        clip_range_vf=cfg.clip_range_vf,
        vf_coef=cfg.vf_coef,
        ent_coef=cfg.ent_coef,
        kl_coef=cfg.kl_coef,
        target_kl=cfg.target_kl,
    )
    reward_cfg = RewardConfig()

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- main loop ---
    # Wrap in try/finally so even if training crashes / is Ctrl-C'd / hits
    # an unhandled exception, we tell the Windows-side bot to stop. This
    # prevents the bot from continuing to call its own LLM (gpt-5-nano)
    # and burn tokens after we've walked away.
    try:
        _run_main_loop(
            cfg, env, policy, optimizer, loss_cfg, reward_cfg, save_dir,
        )
    finally:
        _stop_bot(env)


def _stop_bot(env) -> None:
    """Best-effort: tell the bot to stop self-prompting and abort its
    current action. Called in train_loop's finally so it runs on normal
    completion AND on KeyboardInterrupt / crash.

    Order matters: !stfu first kills the self_prompter (no more LLM
    calls), then !stop aborts whatever skill was running.
    """
    for cmd in ("!stfu", "!stop"):
        try:
            log.info(f"  shutdown: sending {cmd} to bot")
            env.client.execute_command(cmd, timeout=5.0)
        except Exception as e:
            log.warning(f"  shutdown {cmd} failed (bot may already be gone): {e}")


def _run_main_loop(cfg, env, policy, optimizer, loss_cfg, reward_cfg, save_dir):
    for update in range(cfg.n_updates):
        t_update = time.time()

        # 1. collect N episodes — each episode samples a task from the pool
        # (ref policy is implicit: it's the policy at this point — the same
        # one that produced logp_old during rollout. We reuse logp_old as
        # logp_ref in _update_step, avoiding any LoRA swap.)
        #
        # Stratified sampling: every batch contains at least one
        # SIMPLE_TASK (collect_*). Without this, all-hard-task batches
        # produce zero milestone fires and a monotone-bad reward signal
        # that inflates KL (observed in round-2 smoke: 0 ms_fired,
        # kl=0.318 in epoch-0 minibatch-0).
        from rl_spark.task_pool import sample_tasks_stratified

        task_specs_this_update = sample_tasks_stratified(cfg.n_episodes_per_update)
        episodes: list[EpisodeRollout] = []
        episode_tasks: list = []   # parallel list of TaskSpec for logging
        for ep_i in range(cfg.n_episodes_per_update):
            t_ep = time.time()
            # Sample a task for this episode and set up per-task state
            task_spec = task_specs_this_update[ep_i]
            env._pending_goal_cmd = task_spec.goal_template   # consumed by env.reset wrapper
            # Per-task episode budget — env.step also enforces this via truncation
            env.max_steps = task_spec.max_steps
            # Tell SparkScorer about the new task (used for reward calc + soft tokens)
            if hasattr(env, "scorer") and hasattr(env.scorer, "set_task"):
                env.scorer.set_task(task_spec.task_text)

            episode = collect_episode(
                env=env,
                policy=policy,
                value_head=policy.value_head,
                task=task_spec.task_text,
                task_spec=task_spec,
                reward_cfg=reward_cfg,
                max_steps=task_spec.max_steps,   # per-task episode budget
            )
            episodes.append(episode)
            episode_tasks.append(task_spec)
            log.info(
                f"[update {update} ep{ep_i}] task={task_spec.name} "
                f"len={episode.length} total_reward={episode.total_reward():.3f} "
                f"({time.time() - t_ep:.1f}s)"
            )

        # 3. GAE per episode
        advantages_per_ep = []
        returns_per_ep = []
        for ep in episodes:
            rewards = ep.stack("reward").astype(np.float32)
            values = ep.stack("value").astype(np.float32)
            dones = ep.stack("done").astype(np.float32)
            adv, ret = compute_gae(
                rewards, values, dones,
                gamma=cfg.gamma, lam=cfg.lam,
                bootstrap_value=ep.bootstrap_value,
            )
            if cfg.standardize_advantages:
                adv = standardize(adv)
            advantages_per_ep.append(adv)
            returns_per_ep.append(ret)

        # 4. flatten (features_list with PIL frames inside lives on the batch)
        batch = flatten_episodes(
            episodes, advantages_per_ep, returns_per_ep,
            device=torch.device(cfg.device),
        )

        # 5. K PPO epochs
        all_logs = []
        for epoch in range(cfg.n_epochs):
            idx = np.random.permutation(batch.size)
            for start in range(0, len(idx), cfg.minibatch_size):
                mb_idx = idx[start:start + cfg.minibatch_size]
                mb = _slice_flatbatch(batch, mb_idx)
                log_dict = _update_step(
                    policy, mb, optimizer,
                    loss_cfg, cfg,
                )
                all_logs.append(log_dict)

                # Early-stop epoch if KL exceeds target
                if (
                    cfg.target_kl is not None
                    and log_dict.get("ppo/approx_kl_old_new", 0.0) > cfg.target_kl
                ):
                    log.warning(
                        f"early-stop epoch {epoch} at minibatch: "
                        f"kl={log_dict['ppo/approx_kl_old_new']:.4f} > {cfg.target_kl}"
                    )
                    break

        # 6. log + save
        agg = {k: float(np.mean([d[k] for d in all_logs if k in d]))
               for k in {k for d in all_logs for k in d.keys()}}
        agg["update/elapsed_s"] = time.time() - t_update
        agg["update/total_reward"] = float(np.mean([e.total_reward() for e in episodes]))

        # Aggregate per-step reward components across all transitions of
        # this update. Each Transition.reward_log carries the keys produced
        # by assemble_reward (r/delta_weighted, r/aq_weighted,
        # r/len_weighted, r/ms_weighted, etc.). Summing per-episode then
        # averaging gives us a per-episode breakdown of what's driving reward.
        component_keys = (
            "r/delta_weighted",     # SparkScorer Δ (alpha=0.3)
            "r/aq_weighted",        # 2×2 advice quality
            "r/len_weighted",       # length penalty
            "r/ms_weighted",        # milestone reward (main signal)
        )
        for k in component_keys:
            per_ep_sums = []
            for ep in episodes:
                s = 0.0
                for t in ep.transitions:
                    v = t.reward_log.get(k)
                    if isinstance(v, (int, float)):
                        s += float(v)
                per_ep_sums.append(s)
            if per_ep_sums:
                agg[f"reward_sum/{k}"] = float(np.mean(per_ep_sums))

        # Milestone completion stats: how many milestones fired total across
        # episodes, how many episodes hit the final milestone (= task done)
        total_ms_fired = 0
        n_final_hit = 0
        for ep in episodes:
            ep_final_hit = False
            for t in ep.transitions:
                fired = t.reward_log.get("r/ms_fired") or []
                total_ms_fired += len(fired)
                if t.reward_log.get("r/ms_completed_final"):
                    ep_final_hit = True
            if ep_final_hit:
                n_final_hit += 1
        agg["milestone/total_fired"] = float(total_ms_fired)
        agg["milestone/episodes_completed"] = float(n_final_hit)
        agg["milestone/episodes_total"] = float(len(episodes))

        # Per-task breakdown: avg total_reward and completion rate by task
        from collections import defaultdict
        task_buckets: dict[str, list[float]] = defaultdict(list)
        task_completed: dict[str, list[int]] = defaultdict(list)
        for ep, spec in zip(episodes, episode_tasks):
            task_buckets[spec.name].append(ep.total_reward())
            ep_final_hit = any(
                t.reward_log.get("r/ms_completed_final") for t in ep.transitions
            )
            task_completed[spec.name].append(1 if ep_final_hit else 0)
        for name, rewards in task_buckets.items():
            agg[f"task/{name}/reward"] = float(np.mean(rewards))
            agg[f"task/{name}/completion_rate"] = float(np.mean(task_completed[name]))
            agg[f"task/{name}/n_episodes"] = len(rewards)

        log.info(f"[update {update}] {agg}")
        # SPARK hook diagnostic: confirms whether the projector path is
        # actually live. If 'spliced_*' stays 0 across updates, the
        # placeholders aren't being found in input_ids and projector
        # gradient is structurally 0.
        log.info(f"  spark_hook_stats: {policy.spark_hook_stats}")

        if (update + 1) % cfg.save_every_updates == 0:
            ckpt_path = save_dir / f"ppo_update{update:04d}.pt"
            policy.save_trainable(ckpt_path)


def _slice_flatbatch(batch: FlatBatch, idx: np.ndarray) -> FlatBatch:
    """Subset a FlatBatch by integer indices.

    idx is a 1-D np.ndarray of ints. Tensor fields use the ndarray directly
    (PyTorch accepts numpy int arrays as fancy indexers). list[dict] /
    list[Tensor] fields are sliced via list comprehension.
    """
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=batch.advantages.device)
    return FlatBatch(
        features_list=[batch.features_list[i] for i in idx],
        response_ids=[batch.response_ids[i] for i in idx],
        logp_olds=[batch.logp_olds[i] for i in idx],
        advantages=batch.advantages.index_select(0, idx_t),
        returns=batch.returns.index_select(0, idx_t),
        values_old=batch.values_old.index_select(0, idx_t),
        is_help=batch.is_help.index_select(0, idx_t),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> PPOTrainConfig:
    p = argparse.ArgumentParser(description="PPO online training for SPARK agent")
    p.add_argument("--task", default=PPOTrainConfig.task)
    p.add_argument("--n-updates", type=int, default=PPOTrainConfig.n_updates)
    p.add_argument("--n-episodes-per-update", type=int, default=PPOTrainConfig.n_episodes_per_update)
    p.add_argument("--max-steps", type=int, default=PPOTrainConfig.max_steps)
    p.add_argument("--save-dir", default=PPOTrainConfig.save_dir)
    p.add_argument("--spark-trainable-ckpt", default=PPOTrainConfig.spark_trainable_ckpt)
    p.add_argument("--lr-lora", type=float, default=PPOTrainConfig.lr_lora)
    p.add_argument("--seed", type=int, default=PPOTrainConfig.seed)
    p.add_argument("--save-every-updates", type=int, default=PPOTrainConfig.save_every_updates)
    p.add_argument("--resume-from", default=PPOTrainConfig.resume_from,
                   help="Path to a previous ppo_update*.pt to resume LoRA + projector + value_head from")
    a = p.parse_args()
    return PPOTrainConfig(
        task=a.task,
        n_updates=a.n_updates,
        n_episodes_per_update=a.n_episodes_per_update,
        max_steps=a.max_steps,
        save_dir=a.save_dir,
        save_every_updates=a.save_every_updates,
        spark_trainable_ckpt=a.spark_trainable_ckpt,
        lr_lora=a.lr_lora,
        seed=a.seed,
        resume_from=a.resume_from,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    cfg = parse_args()
    train_loop(cfg)


if __name__ == "__main__":
    main()
