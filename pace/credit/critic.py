"""
critic.py — CompanionWithValueHead: LM head + Value head on shared Qwen backbone.

Design:
  - Base: Qwen3.5-VL-9B, frozen (via HF default)
  - LoRA (r=16) via PEFT: attention projections, trainable
  - LM head: inherits from base (frozen)
  - Value head: 2-layer MLP (Linear→GELU→Linear), trainable
      Init last layer to zero → V(s) ≈ 0 at start (safe cold start)

  Forward pass: one call through Qwen produces both:
    - logits: (batch, seq_len, vocab_size) — for LM / PPO
    - value:  (batch,)                     — pooled from last-token hidden

  Pooling strategy: last non-pad token's hidden state (causal LM standard).

Usage:
  model = CompanionWithValueHead.from_pretrained(
      base_model_path="/workspace/models/Qwen3.5-9B",
      adapter_path="/workspace/checkpoints/companion_sft_v3_C",
  )
  logits, value = model(input_ids, attention_mask, return_logits=True)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    """Small MLP: hidden_state (4096) → value (scalar).

    Two layers with GELU. Last layer init to zero so V(s) ≈ 0 at start —
    avoids critic pushing PPO in wrong direction before it's trained.
    """

    def __init__(self, hidden_dim: int, mid_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_hidden).squeeze(-1)


class CompanionWithValueHead(nn.Module):
    """Companion model: shared Qwen backbone + LM head (frozen) + Value head (train).

    LoRA adapter comes from SFT-C checkpoint and is loaded as trainable.
    All other Qwen parameters are frozen.
    """

    def __init__(self, qwen_model, value_head: ValueHead, processor=None):
        super().__init__()
        self.qwen = qwen_model
        self.value_head = value_head
        self.processor = processor      # AutoProcessor for VL; None for text-only

    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        adapter_path: Optional[str] = None,
        value_head_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        vision: bool = False,
        create_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_targets: Optional[list] = None,
    ) -> "CompanionWithValueHead":
        """Load Qwen base + LoRA adapter (if given) + value head (if given).

        Args:
          base_model_path: /workspace/models/Qwen3.5-9B
          adapter_path:    /workspace/checkpoints/companion_sft_v3_C  (LoRA)
          value_head_path: if training from scratch, leave None.
                           If continuing / evaluating, path to value_head.pt.
          vision:          if True, load the full VL model (Siglip2 vision encoder
                           + language model) via AutoModelForImageTextToText plus an
                           AutoProcessor, so the companion can be trained on frames
                           (Option A / A2). If False, load the text-only CausalLM
                           (ALFWorld / Option B). The vision ENCODER stays frozen;
                           only the language-layer LoRA + value head train.
        """
        processor = None
        if vision:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            print(f"[critic] loading VL model from {base_model_path}", flush=True)
            base = AutoModelForImageTextToText.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        else:
            from transformers import AutoModelForCausalLM
            print(f"[critic] loading Qwen (text) from {base_model_path}", flush=True)
            base = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation="eager",
            )
        base.config.use_cache = False        # required for training
        # Enable gradient checkpointing to halve activation memory
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("[critic] gradient checkpointing enabled", flush=True)
        except Exception as e:
            print(f"[critic] gradient_checkpointing_enable failed: {e}", flush=True)

        if adapter_path is not None:
            print(f"[critic] loading LoRA adapter from {adapter_path} (trainable)",
                  flush=True)
            from peft import PeftModel
            qwen_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
            # PEFT sometimes disables input_requires_grad on frozen base;
            # this call re-enables it so backprop through the frozen layers works
            if hasattr(qwen_model, "enable_input_require_grads"):
                qwen_model.enable_input_require_grads()
        elif create_lora:
            # Fresh zero-init LoRA on the base (VL) model: the companion starts
            # exactly as the base VL (grounded), and NIC trains this new adapter.
            # Targets the attention projections of the LANGUAGE layers; the
            # vision encoder stays frozen (no LoRA on Siglip2).
            from peft import LoraConfig, get_peft_model
            targets = lora_targets or ["q_proj", "k_proj", "v_proj", "o_proj"]
            print(f"[critic] creating FRESH LoRA (r={lora_r}, targets={targets}) on base",
                  flush=True)
            lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                target_modules=targets, task_type="CAUSAL_LM", bias="none",
            )
            qwen_model = get_peft_model(base, lora_cfg)
            if hasattr(qwen_model, "enable_input_require_grads"):
                qwen_model.enable_input_require_grads()
            try:
                qwen_model.print_trainable_parameters()
            except Exception:
                pass
        else:
            qwen_model = base

        # Value head placed on same device as backbone's last layer.
        # Kept in fp32 (small, ~2M params) — avoids bf16↔fp32 grad dtype
        # mismatches during loss.backward(); the fp32 grads flow back through
        # a cast to bf16 for the backbone hidden state input.
        # VL configs nest the LM dims under text_config.
        hidden_dim = getattr(base.config, "hidden_size", None)
        if hidden_dim is None:
            hidden_dim = base.config.text_config.hidden_size
        value_head = ValueHead(hidden_dim=hidden_dim, mid_dim=512)
        try:
            last_layer_device = next(reversed(list(base.parameters()))).device
        except Exception:
            last_layer_device = torch.device("cuda:0")
        value_head = value_head.to(last_layer_device)   # keep fp32 default
        print(f"[critic] value head initialized on {last_layer_device} "
              f"(fp32)", flush=True)

        # Optionally load pre-trained value head
        if value_head_path is not None and os.path.isfile(value_head_path):
            sd = torch.load(value_head_path, map_location=last_layer_device)
            value_head.load_state_dict(sd)
            print(f"[critic] value head loaded from {value_head_path}", flush=True)

        return cls(qwen_model=qwen_model, value_head=value_head, processor=processor)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Forward pass. Returns:
          - logits: (batch, seq_len, vocab) if return_logits, else None
          - value:  (batch,)

        For the VL companion, pass `pixel_values` + `image_grid_thw` (produced by
        the processor from the frames). They are ignored by a text-only backbone.
        """
        qwen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        if pixel_values is not None:
            qwen_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            qwen_kwargs["image_grid_thw"] = image_grid_thw
        if mm_token_type_ids is not None:
            qwen_kwargs["mm_token_type_ids"] = mm_token_type_ids
        outputs = self.qwen(**qwen_kwargs)
        last_hidden = outputs.hidden_states[-1]      # (B, L, D)

        # Pool: last non-pad token
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
            pooled = last_hidden[batch_idx, seq_lens]
        else:
            pooled = last_hidden[:, -1, :]

        # Move pooled to value_head's device
        vh_device = next(self.value_head.parameters()).device
        if pooled.device != vh_device:
            pooled = pooled.to(vh_device)
        # Cast pooled to fp32 (value head is fp32); autograd will cast the
        # gradient back to bf16 when it flows into the backbone hidden state.
        pooled_fp32 = pooled.to(torch.float32)

        value = self.value_head(pooled_fp32)

        return (outputs.logits, value) if return_logits else (None, value)

    def save(self, output_dir: str) -> None:
        """Save LoRA adapter and value head separately.

        Layout:
          output_dir/
            adapter_config.json          ← PEFT LoRA
            adapter_model.safetensors
            value_head.pt                ← nn.Module state_dict
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.qwen.save_pretrained(output_dir)
        vh_path = os.path.join(output_dir, "value_head.pt")
        torch.save(self.value_head.state_dict(), vh_path)
        print(f"[critic] saved LoRA to {output_dir} and value head to {vh_path}",
              flush=True)

    def trainable_params_by_group(self) -> dict:
        """Return dict of {group_name: [params]} for AdamW with per-group LR."""
        lora_params, value_params, other = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "value_head" in name:
                value_params.append(p)
            elif "lora_" in name.lower():
                lora_params.append(p)
            else:
                other.append(p)
        return {"lora": lora_params, "value_head": value_params, "other": other}
