# PACE — Proactive Assistance through nested Credit Estimation

Reference code for the ICLR submission *PACE: Proactive Assistance through
nested Credit Estimation*. A trainable vision–language **companion**
(Qwen-VL backbone + a single LoRA) learns **when** to intervene (the *gate*)
and **what** advice to give (the *content*) for a **frozen** advisee acting in
ALFWorld (text) and Minecraft (vision).

This is an anonymized, trimmed snapshot for review. It is meant to be **read**,
not run end-to-end: server addresses, checkpoints, dataset paths and some
internal utilities are omitted, and cross-references to `method.md` /
`docs/interfaces.md` in the docstrings point at the (non-anonymized) design
notes and are kept only as pointers to the paper's sections.

## What maps to what in the paper

**Nested credit estimation (Method, §4).** The companion is one model +
one LoRA emitting a single `{"gate": ..., "advice": ...}` JSON. Gate and
content credit are separated by token masks and different advantages, then
optimized in a two-pass PPO update.

- `pace/credit/brancher.py` — matched interventional branches from a
  replayable state (silence / replay / K milestone-focused fresh) under
  common random numbers (CRN). This is the counterfactual sampler.
- `pace/ppo/cf_expander.py` — expands each HELP state into the K+2 branches.
- `pace/ppo/cf_nstep_q.py` — N-step return with critic tail bootstrap
  `V_{φ^-}` for each branch (the branch value `Q_i`).
- `pace/ppo/cf_advantage.py` — the two levels of credit: gate credit
  `Δ_t` (replay-help vs silence) and content credit
  `Â_{t,i} = (Q_i − Q̄)/σ` over the help branches.
- `pace/ppo/cf_token_masks.py` — locates the gate-value vs advice-value token
  spans inside the JSON so each gets its own advantage.
- `pace/ppo/cf_gate_trainer.py`, `pace/ppo/cf_content_trainer.py`
  (and `_vl.py` vision variants) — the two-pass content-then-gate PPO update.
- `pace/ppo/cf_reward_assembler.py` — assembles the step reward
  (task success + milestone bonuses − HELP cost).
- `pace/credit/critic.py`, `pace/ppo/value_head.py` — value head (2-layer MLP,
  zero-init) and critic bookkeeping.
- `pace/credit/reward.py` — reward definition (Eq. 2): milestone-first-fire
  bonus, terminal outcome, and the recency-shaped per-intervention cost.
- `pace/credit/dr_estimator.py`, `pace/credit/propensity_log.py` — off-policy /
  propensity utilities used by the estimator and probes.

**Environments (Experimental setup).**

- `pace/envs/alfworld_env.py` — ALFWorld advisee env with replayable state and
  intermediate milestones (text).
- `pace/envs/mc_env.py`, `pace/envs/mindcraft_env.py` — Minecraft advisee env
  (vision; command-line reset ≈ approximate replay), per-task budgets and
  item-specific milestone coefficients.
- `pace/envs/task_pool.py` — task specifications and milestone chains.

**Warm-start & training entry points.**

- `pace/scripts/gen_sft_teacher_v3.py` — generate teacher `{gate, advice}`
  data for advice SFT.
- `pace/scripts/train_sft_companion.py` — completion-only advice SFT.
- `pace/scripts/train_critic_warmup.py` — critic warm-up (regress to
  discounted return-to-go, no bootstrap).
- `pace/scripts/train_cf_grpo.py` — ALFWorld main training loop.
- `pace/scripts/train_mc.py` — Minecraft main training loop.
- `pace/scripts/mc_train_pipeline.sh` — end-to-end Minecraft pipeline.

**Evaluation & live study.**

- `pace/scripts/mc_eval.py`, `pace/scripts/offline_eval.py` — evaluation
  (includes the fixed-step / random-step gate baselines).
- `pace/live/coach.py` — live human-study coach (3 tasks × conditions).
- `pace/live/stream_screen.py` — frame streamer for the vision companion.

**Prompts.** `pace/prompts/` — companion / advisee prompt builders for
ALFWorld and Minecraft (`*_live.py` = human-facing advice variants).

## Layout

```
pace/
  advisee.py            frozen advisee wrapper
  policy.py  utils.py   shared model / helpers
  envs/                 ALFWorld & Minecraft envs, task pool
  credit/               reward, brancher, critic, estimators
  ppo/                  nested-credit two-pass PPO (cf_*.py)
  prompts/              prompt builders
  scripts/              warm-start, training, evaluation
  live/                 live human-study coach + frame streamer
```
