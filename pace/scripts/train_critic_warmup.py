"""
train_critic_warmup.py — supervised MSE warmup for the value head.

Data: rollouts_combined.jsonl (from collect_rollouts.py). Each line is a step
with a `G` field (MC discounted return under our reward function).

Target: value head predicts G given the state prompt (system + user, no
assistant response). LoRA + value head both trainable; backbone frozen.

Why we do this: PPO with a random-init critic wastes 50-100 updates cold-
starting the value function. Warming up on measured returns from SFT-C
rollouts (independent of any prior SPARK Score data) gives PPO an
informative baseline from step 0.

Usage:
  cd /workspace/mindcraft
  PYTHONPATH=. python3 rl_causal/scripts/train_critic_warmup.py \\
      --model /workspace/models/Qwen3.5-9B \\
      --adapter /workspace/checkpoints/companion_sft_v3_C \\
      --data /workspace/sft_data/rollouts_combined.jsonl \\
      --output-dir /workspace/checkpoints/companion_sft_v3_C_with_critic \\
      --epochs 2 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from rl_causal.critic import CompanionWithValueHead
from rl_causal.prompts.alfworld import (
    SYSTEM_PROMPT_COMPANION,
    build_companion_prompt,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_records(jsonl_path: str) -> list:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_state_prompt(record: dict) -> tuple:
    """Return (chat messages list, target_V float) for a single record."""
    obs = {
        "task_description":     record.get("task_desc", ""),
        "text":                 record.get("state_text", ""),
        "inventory_text":       record.get("inventory_text", ""),
        "admissible_commands":  record.get("admissible", []),
    }
    history = record.get("history", []) or []
    user_prompt = build_companion_prompt(obs, history=history)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COMPANION},
        {"role": "user",   "content": user_prompt},
    ]
    return messages, float(record["G"])


def batch_tokenize(tokenizer, records: list, max_len: int, device: torch.device):
    """Tokenize a batch of records (state prompt only, no assistant response).

    Returns:
      input_ids     (B, L)
      attention_mask (B, L)
      targets       (B,)  as torch.float
    """
    rendered = []
    targets = []
    for r in records:
        msgs, G = build_state_prompt(r)
        try:
            text = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
            )
        rendered.append(text)
        targets.append(G)

    enc = tokenizer(
        rendered,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    targets_t = torch.tensor(targets, dtype=torch.float32, device=device)
    return input_ids, attention_mask, targets_t


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tokenizer, records, batch_size, max_len, device):
    model.eval()
    preds, targets = [], []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        input_ids, attn, T = batch_tokenize(tokenizer, batch, max_len, device)
        _, V = model(input_ids, attention_mask=attn, return_logits=False)
        preds.extend(V.detach().float().cpu().tolist())
        targets.extend(T.detach().float().cpu().tolist())
    model.train()

    import statistics as st
    mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / max(1, len(preds))
    # Pearson correlation
    n = len(preds)
    if n < 2:
        pearson = float("nan")
    else:
        mp, mt = sum(preds) / n, sum(targets) / n
        num = sum((p - mp) * (t - mt) for p, t in zip(preds, targets))
        den_p = math.sqrt(sum((p - mp) ** 2 for p in preds))
        den_t = math.sqrt(sum((t - mt) ** 2 for t in targets))
        pearson = num / (den_p * den_t) if den_p * den_t > 0 else float("nan")
    return {
        "mse":        mse,
        "pearson_r":  pearson,
        "pred_mean":  st.mean(preds) if preds else 0,
        "pred_std":   st.pstdev(preds) if len(preds) > 1 else 0,
        "target_mean": st.mean(targets) if targets else 0,
        "target_std":  st.pstdev(targets) if len(targets) > 1 else 0,
        "n":          len(preds),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base Qwen path")
    ap.add_argument("--adapter", required=True,
                    help="SFT-C LoRA adapter path (loaded as trainable)")
    ap.add_argument("--data", required=True, help="rollouts JSONL")
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="effective batch = batch_size × grad_accum")
    ap.add_argument("--lr-lora", type=float, default=1e-5)
    ap.add_argument("--lr-value", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="Fraction of samples held out for eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every-epoch", action="store_true", default=True)
    args = ap.parse_args()

    # Preflight
    if not os.path.exists(args.data):
        print(f"ERROR: data not found: {args.data}")
        return 2
    if not os.path.isdir(args.model):
        print(f"ERROR: model dir not found: {args.model}")
        return 2

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Critic warmup")
    print("=" * 70)
    print(f"  model      : {args.model}")
    print(f"  adapter    : {args.adapter}")
    print(f"  data       : {args.data}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  epochs     : {args.epochs}")
    print(f"  batch_size : {args.batch_size} × grad_accum {args.grad_accum} "
          f"= {args.batch_size * args.grad_accum} effective")
    print(f"  LR LoRA    : {args.lr_lora}   LR value: {args.lr_value}")
    print(f"  max_seq    : {args.max_seq_length}")
    print("=" * 70, flush=True)

    # Load data
    records = load_records(args.data)
    random.Random(args.seed).shuffle(records)
    n_val = max(50, int(len(records) * args.val_frac))
    val = records[:n_val]
    train = records[n_val:]
    print(f"[data] train={len(train)}  val={len(val)}", flush=True)

    # Load model
    from transformers import AutoTokenizer
    print(f"[tok] loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = CompanionWithValueHead.from_pretrained(
        base_model_path=args.model,
        adapter_path=args.adapter,
    )
    device = next(model.value_head.parameters()).device

    # Confirm what's trainable
    groups = model.trainable_params_by_group()
    n_lora = sum(p.numel() for p in groups["lora"])
    n_value = sum(p.numel() for p in groups["value_head"])
    print(f"[params] trainable: LoRA={n_lora:,}  value_head={n_value:,}  "
          f"other={sum(p.numel() for p in groups['other']):,}", flush=True)

    # Optimizer with per-group LR
    optimizer = torch.optim.AdamW([
        {"params": groups["lora"],       "lr": args.lr_lora},
        {"params": groups["value_head"], "lr": args.lr_value},
    ], weight_decay=args.weight_decay)

    # Cosine LR schedule
    total_steps = (len(train) // args.batch_size + 1) * args.epochs // args.grad_accum
    total_steps = max(1, total_steps)
    warmup_steps = max(1, int(0.03 * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        # cosine
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initial eval (before any training — should be near-zero V_pred)
    print("[eval] initial (pre-warmup) baseline:", flush=True)
    m0 = evaluate(model, tokenizer, val, args.batch_size, args.max_seq_length, device)
    print(f"  MSE={m0['mse']:.4f}  Pearson_r={m0['pearson_r']:.3f}  "
          f"pred_mean={m0['pred_mean']:.3f}  pred_std={m0['pred_std']:.3f}  "
          f"target_mean={m0['target_mean']:.3f}  target_std={m0['target_std']:.3f}",
          flush=True)

    # Training loop
    print(f"[train] starting; total_steps={total_steps}  warmup={warmup_steps}",
          flush=True)
    global_step = 0
    t_start = time.time()
    running_loss = 0.0
    running_n = 0

    for epoch in range(args.epochs):
        random.shuffle(train)

        # Iterate over mini-batches; grad_accum
        for i in range(0, len(train), args.batch_size):
            batch = train[i:i + args.batch_size]
            if len(batch) == 0:
                continue
            input_ids, attn, targets = batch_tokenize(
                tokenizer, batch, args.max_seq_length, device,
            )

            _, V_pred = model(input_ids, attention_mask=attn, return_logits=False)
            # V_pred is fp32 (value_head is fp32), targets is fp32 → clean MSE
            loss = F.mse_loss(V_pred, targets)
            loss = loss / args.grad_accum
            loss.backward()

            running_loss += loss.item() * args.grad_accum
            running_n += 1

            if running_n % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in groups.values() for p in g],
                    max_norm=args.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    lr_lora_cur = optimizer.param_groups[0]["lr"]
                    lr_val_cur = optimizer.param_groups[1]["lr"]
                    dt = time.time() - t_start
                    print(f"  step={global_step:4d}/{total_steps}  "
                          f"loss={running_loss/max(1,running_n):.4f}  "
                          f"lr_lora={lr_lora_cur:.2e}  lr_val={lr_val_cur:.2e}  "
                          f"{dt:.0f}s elapsed", flush=True)
                    running_loss, running_n = 0.0, 0

        # End-of-epoch eval
        print(f"[eval] epoch {epoch+1}/{args.epochs}", flush=True)
        m = evaluate(model, tokenizer, val, args.batch_size, args.max_seq_length, device)
        print(f"  MSE={m['mse']:.4f}  Pearson_r={m['pearson_r']:.3f}  "
              f"pred_std={m['pred_std']:.3f}  target_std={m['target_std']:.3f}",
              flush=True)

    # Final save
    print(f"[save] writing to {args.output_dir}", flush=True)
    model.save(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Final eval summary
    print()
    print("=" * 70)
    print("Critic warmup DONE")
    print("=" * 70)
    m_final = evaluate(model, tokenizer, val, args.batch_size, args.max_seq_length, device)
    print(f"  final MSE          : {m_final['mse']:.4f}")
    print(f"  final Pearson r    : {m_final['pearson_r']:.3f}")
    print(f"  V_pred distribution: mean={m_final['pred_mean']:.3f}  "
          f"std={m_final['pred_std']:.3f}")
    print(f"  Target distribution: mean={m_final['target_mean']:.3f}  "
          f"std={m_final['target_std']:.3f}")
    print(f"  Val samples        : {m_final['n']}")
    print()
    print("Sanity thresholds:")
    print(f"  Pearson r ≥ 0.5  :  {'PASS' if m_final['pearson_r'] >= 0.5 else 'FAIL'}")
    print(f"  MSE       ≤ 0.15 :  {'PASS' if m_final['mse']       <= 0.15 else 'FAIL'}")
    print(f"  pred_std  ≥ 0.3  :  {'PASS' if m_final['pred_std']  >= 0.3  else 'FAIL'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
