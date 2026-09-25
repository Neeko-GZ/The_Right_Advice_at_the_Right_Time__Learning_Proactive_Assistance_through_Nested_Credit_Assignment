"""rl_causal.ppo — PPO trainer subpackage.

Layout (SPARK CF-nstep-GRPO, from W31 refactor):

Foundation (single-file utilities, independently testable):
    cf_token_masks.py      locate advice/gate value tokens in JSON output
    cf_nstep_q.py          n-step Q with critic tail bootstrap
    cf_reward_assembler.py intrinsic quality + intra-group diversity

Rollout + counterfactual (needs env + companion):
    cf_rollout_worker.py   parallel env rollout with log_π + snapshot
    cf_expander.py         K+2 branch counterfactual expansion per HELP state

Advantage + loss:
    cf_advantage.py        K+1 group-relative advantage
    cf_content_trainer.py  content-side PPO loop (mask to advice tokens)
    cf_gate_trainer.py     gate-side PPO loop (mask to gate token)

Entry:
    (see rl_causal/scripts/train_cf_grpo.py)

Legacy (SPARK v1, kept for reference until v2 stable):
    value_head.py, rollout.py, advantages.py, losses.py, train.py

Design reference: md/SPARK_finetune_unified.md, method.md § 4.2, § 5.5.
"""
