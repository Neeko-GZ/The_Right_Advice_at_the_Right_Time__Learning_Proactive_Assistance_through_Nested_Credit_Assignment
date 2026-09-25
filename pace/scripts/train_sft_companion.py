"""
train_sft_companion.py — SFT warm-start for the companion.

Trains Qwen3.5-VL-9B (loaded but vision-encoder frozen) with LoRA rank 16
on companion_teacher_v2.jsonl. Companion learns: given (task, state,
admissible, history) → produce natural-language advice.

Design (aligned with method.md § 5.7 A + § 5.6):
  - Backbone: Qwen3.5-VL-9B loaded as VL model; vision encoder frozen.
  - LoRA: rank 16, target text-side attention only (q/k/v/o_proj).
  - Objective: teacher-forcing on advice tokens only (completion_only_loss).
  - Loss mask: system + user prompt is masked; only advice tokens contribute.
  - Chat template: Qwen native (system/user/assistant roles).
  - Prompt template must match VllmAdvisee's inference-time expectations
    (task, history, current observation, inventory, admissible) so the
    trained companion's output is directly usable in advisee's advice slot.

Usage:
  # Dry-run: 100 samples, 1 epoch, small batch (5-10 min)
  cd /workspace/mindcraft
  PYTHONPATH=. python3 rl_causal/scripts/train_sft_companion.py \\
      --data /workspace/sft_data/companion_teacher_v2.jsonl \\
      --model /workspace/models/Qwen3.5-9B \\
      --output-dir /workspace/checkpoints/companion_sft_dryrun \\
      --dry-run

  # Full training (background, ~6-15 h with packing)
  cd /workspace/mindcraft
  nohup env PYTHONPATH=. python3 -u rl_causal/scripts/train_sft_companion.py \\
      --data /workspace/sft_data/companion_teacher_v2.jsonl \\
      --model /workspace/models/Qwen3.5-9B \\
      --output-dir /workspace/checkpoints/companion_sft_v1 \\
      > /tmp/train_sft.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Prompt template — imported from rl_causal.prompts.alfworld
# (Phase 2 MC will swap to rl_causal.prompts.mc)
# ---------------------------------------------------------------------------

from rl_causal.prompts.alfworld import (
    SYSTEM_PROMPT_COMPANION as SYSTEM_PROMPT,
    build_companion_prompt,
)


def _format_user_prompt(record: dict) -> str:
    """Adapter: JSONL record → dict → build_companion_prompt.

    JSONL records use `state_text` / `admissible` / `inventory_text` keys;
    prompt builder expects an obs dict with `text` / `admissible_commands` /
    `inventory_text`. Convert here.
    """
    obs = {
        "task_description": record["task_desc"],
        "text":             record["state_text"],
        "inventory_text":   record.get("inventory_text", ""),
        "admissible_commands": record.get("admissible", []),
    }
    history = record.get("history", None)
    return build_companion_prompt(obs, history=history)


def _format_assistant_output(record: dict) -> str:
    """Format the assistant target string.

    v4-onwards: JSON {"gate": "HELP"|"SILENCE", "advice": "..."}.
    Backward-compat with v2/v3-legacy records (no `gate` field): infer
    gate=HELP + advice=record['advice'].
    """
    gate = record.get("gate")
    if gate is None:
        # Legacy: no gate field → assume HELP + plain advice
        gate = "HELP"
        advice = record.get("advice", "")
    else:
        gate = str(gate).upper()
        advice = record.get("advice", "") if gate == "HELP" else ""
    return json.dumps({"gate": gate, "advice": advice}, ensure_ascii=False)


def _record_to_messages(record: dict) -> dict:
    """Convert a JSONL record → chat-template messages dict.
    SFTTrainer (trl >= 0.29) auto-detects the 'messages' key and applies
    the tokenizer's chat template with completion_only_loss."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_user_prompt(record)},
            {"role": "assistant", "content": _format_assistant_output(record)},
        ]
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(jsonl_path: str, limit: Optional[int] = None):
    """Load JSONL → HF Dataset of {'messages': [...]} records."""
    from datasets import Dataset

    records = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit is not None and len(records) >= limit:
                break
            r = json.loads(line)
            records.append(_record_to_messages(r))
    print(f"[dataset] loaded {len(records)} records from {jsonl_path}")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Model loading + LoRA + vision-freeze
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_path: str, bf16: bool = True):
    """Load Qwen3.5-VL-9B and its tokenizer. Vision encoder frozen later
    via named_parameters iteration."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"[model] loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[model] loading Qwen3.5-VL-9B from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",   # safer than flash for LoRA + custom arch
    )
    model.config.use_cache = False   # required for gradient checkpointing / LoRA training
    return model, tokenizer


def _freeze_vision(model) -> tuple:
    """Freeze all vision-related parameters. Returns (n_frozen_params, n_kept)."""
    n_frozen = 0
    n_kept = 0
    for name, param in model.named_parameters():
        lname = name.lower()
        is_vision = any(k in lname for k in
                        ["vision", "visual", "image", "vit", "patch_embed"])
        if is_vision:
            param.requires_grad = False
            n_frozen += param.numel()
        else:
            n_kept += param.numel()
    print(f"[freeze] vision params frozen: {n_frozen:,}   "
          f"language params still active: {n_kept:,}")
    return n_frozen, n_kept


def _apply_lora(model, rank: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Apply LoRA to text-side attention only.

    target_modules uses exact projection names (q_proj/k_proj/v_proj/o_proj).
    peft matches by module name suffix, so this hits attention throughout the
    model. To avoid touching vision-side attention, we pass modules_to_save=None
    and rely on requires_grad=False set by _freeze_vision (LoRA respects
    parent module's requires_grad).
    """
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _resume_from_adapter(model, adapter_path: str):
    """SFT-B path: load existing LoRA adapter (SFT-A) and keep training it.

    Unlike _apply_lora (fresh LoRA), this attaches the pre-trained adapter
    and marks its weights as trainable so the optimizer can update them.
    Use with small LR (1e-5) since we're doing style transfer, not from scratch.
    """
    from peft import PeftModel

    print(f"[resume] loading LoRA adapter from {adapter_path} for continued training")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    # Explicitly ensure LoRA params are trainable (PeftModel default is True
    # when is_trainable=True but double-check).
    n_trainable = 0
    for name, p in model.named_parameters():
        if "lora_" in name.lower():
            p.requires_grad = True
            n_trainable += p.numel()
    print(f"[resume] LoRA trainable params: {n_trainable:,}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to SFT teacher JSONL")
    ap.add_argument("--model", required=True, help="Path to Qwen3.5-VL-9B")
    ap.add_argument("--output-dir", required=True, help="Where to save LoRA + logs")

    # Training hyperparams (method.md § 5.6 defaults)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="Per-device train batch size")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="Grad accum steps → effective batch = batch_size * grad_accum")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-seq-length", type=int, default=2048,
                    help="Data p95 total = ~720 tokens; max = ~1650; 2048 has buffer.")

    # Efficiency + logging
    ap.add_argument("--packing", action="store_true",
                    help="Enable sequence packing for 3-5x speedup (recommended)")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--save-total-limit", type=int, default=3)

    # SFT-B continue-training from SFT-A adapter
    ap.add_argument("--resume-from-adapter", default=None,
                    help="Path to existing LoRA adapter (e.g. SFT-A checkpoint). "
                         "If set, resumes training that adapter instead of "
                         "creating a fresh one. Use with small LR (1e-5).")

    # Debugging
    ap.add_argument("--dry-run", action="store_true",
                    help="Sanity: 100 samples, 1 epoch, small batch")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of samples (for dry-run / debug)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dry_run:
        args.limit = args.limit or 100
        args.epochs = 1.0
        args.save_steps = 50
        args.logging_steps = 2
        args.output_dir = args.output_dir + "_dryrun"

    # Preflight
    if not os.path.exists(args.data):
        print(f"ERROR: data file not found: {args.data}")
        return 2
    if not os.path.isdir(args.model):
        print(f"ERROR: model dir not found: {args.model}")
        return 2

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SFT companion training")
    print("=" * 70)
    print(f"  data           : {args.data}")
    print(f"  model          : {args.model}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  epochs         : {args.epochs}")
    print(f"  batch_size     : {args.batch_size} × grad_accum {args.grad_accum} "
          f"= effective {args.batch_size * args.grad_accum}")
    print(f"  lr             : {args.lr}   warmup {args.warmup_ratio}")
    print(f"  LoRA           : rank {args.lora_rank}, alpha {args.lora_alpha}")
    print(f"  max_seq_length : {args.max_seq_length}")
    print(f"  packing        : {args.packing}")
    print(f"  dry_run        : {args.dry_run}")
    if args.limit:
        print(f"  limit          : {args.limit}")
    print("=" * 70, flush=True)

    # Load & prep
    dataset = _load_dataset(args.data, limit=args.limit)
    model, tokenizer = _load_model_and_tokenizer(args.model, bf16=True)

    n_frozen, n_kept = _freeze_vision(model)
    if args.resume_from_adapter:
        model = _resume_from_adapter(model, args.resume_from_adapter)
    else:
        model = _apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha,
                            dropout=args.lora_dropout)

    # Train
    from trl import SFTTrainer, SFTConfig

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_length=args.max_seq_length,
        packing=args.packing,
        completion_only_loss=True,     # ★ only train on assistant tokens
        report_to="none",              # disable wandb/tensorboard by default
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"[train] starting; ~{len(dataset)} samples × "
          f"{args.epochs} epochs / effective batch "
          f"{args.batch_size * args.grad_accum} = "
          f"~{int(len(dataset) * args.epochs / (args.batch_size * args.grad_accum))} steps")
    trainer.train()

    # Save final
    print(f"[save] writing final LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print()
    print("=" * 70)
    print("SFT training DONE")
    print(f"  LoRA adapter at: {args.output_dir}")
    print(f"  Next: load with `PeftModel.from_pretrained(base, {args.output_dir!r})`")
    print("  Sanity: generate advice on a held-out state, eyeball quality")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
