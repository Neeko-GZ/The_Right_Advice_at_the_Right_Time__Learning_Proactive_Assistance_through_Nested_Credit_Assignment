"""
cf_gate_trainer_vl.py — VL (A2) gate-side PPO trainer.

Forked from cf_gate_trainer.py (ALFWorld gate trainer stays UNTOUCHED). Mirrors
ContentTrainerVL but:
  * mask = gate token(s) instead of advice tokens
  * advantage = scalar gate_advantage per HELP state (not per branch)
  * within-update KL early-stop (target_kl) as in the text gate trainer
Reuses ppo_clip_loss / gather_log_probs from cf_content_trainer.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from rl_causal.ppo.cf_token_masks import build_token_mask, TokenMask
from rl_causal.ppo.cf_content_trainer import ppo_clip_loss, gather_log_probs
from rl_causal.ppo.cf_content_trainer_vl import _decode_frames


@dataclass
class GateSampleVL:
    """One HELP state's VL gate sample (uses the HELP-replay branch response)."""
    prompt_text: str
    response_text: str
    gate_advantage: float
    frames_b64: List[str] = field(default_factory=list)
    system_prompt: str = ""
    state_id: int = -1


class GateTrainerVL:
    """PPO clip trainer for the gate token with a VL backbone. Mini-batch=1."""

    def __init__(
        self,
        model,                       # CompanionWithValueHead (vision=True)
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_lora: float = 1e-5,
        lr_value: float = 5e-5,
        ppo_clip: float = 0.2,
        target_kl: float = 0.2,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 2,
        max_seq_length: int = 4096,
        device: Optional[torch.device] = None,
    ) -> None:
        assert getattr(model, "processor", None) is not None, \
            "GateTrainerVL needs a VL model (vision=True, has .processor)"
        self.model = model
        self.processor = model.processor
        self.tok = model.processor.tokenizer
        self.ppo_clip = ppo_clip
        self.target_kl = target_kl
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.max_seq_length = max_seq_length
        self.device = device or next(model.parameters()).device

        if optimizer is None:
            pg = model.trainable_params_by_group()
            self.optimizer = torch.optim.AdamW([
                {"params": pg["lora"], "lr": lr_lora},
                {"params": pg["value_head"], "lr": lr_value},
            ])
        else:
            self.optimizer = optimizer

    def _tokenize(self, sample: GateSampleVL) -> Optional[Dict[str, Any]]:
        imgs = _decode_frames(sample.frames_b64)
        user_content = [{"type": "image"} for _ in imgs]
        user_content.append({"type": "text", "text": sample.prompt_text})
        messages = [
            {"role": "system", "content": sample.system_prompt or ""},
            {"role": "user", "content": user_content},
        ]
        prompt_str = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        proc = self.processor(text=[prompt_str], images=imgs if imgs else None,
                              return_tensors="pt")
        prompt_ids = proc["input_ids"][0]
        response_ids = self.tok(sample.response_text, add_special_tokens=False)["input_ids"]
        prompt_len = int(prompt_ids.shape[0])
        full_ids = torch.cat(
            [prompt_ids, torch.tensor(response_ids, dtype=prompt_ids.dtype)], dim=0
        )
        if full_ids.shape[0] > self.max_seq_length:
            return None
        mask = torch.zeros(full_ids.shape[0], dtype=torch.bool)
        tm: TokenMask = build_token_mask(self.tok, sample.response_text, response_ids)
        for local_idx in tm.gate_positions:
            fi = prompt_len + local_idx
            if 0 <= fi < mask.shape[0]:
                mask[fi] = True
        if not bool(mask.any()):
            return None
        mm_full = None
        mm_prompt = proc.get("mm_token_type_ids")
        if mm_prompt is not None:
            pm = mm_prompt[0]
            mm_full = torch.cat(
                [pm, torch.zeros(len(response_ids), dtype=pm.dtype)], dim=0
            ).unsqueeze(0)
        return {
            "input_ids": full_ids.unsqueeze(0),
            "attention_mask": torch.ones(1, full_ids.shape[0], dtype=torch.long),
            "gate_mask": mask.unsqueeze(0),
            "pixel_values": proc.get("pixel_values"),
            "image_grid_thw": proc.get("image_grid_thw"),
            "mm_token_type_ids": mm_full,
        }

    def _forward(self, tok: Dict[str, Any]):
        input_ids = tok["input_ids"].to(self.device)
        attn = tok["attention_mask"].to(self.device)
        pv = tok["pixel_values"].to(self.device) if tok.get("pixel_values") is not None else None
        thw = tok["image_grid_thw"].to(self.device) if tok.get("image_grid_thw") is not None else None
        mm = tok["mm_token_type_ids"].to(self.device) if tok.get("mm_token_type_ids") is not None else None
        logits, _ = self.model(
            input_ids=input_ids, attention_mask=attn, return_logits=True,
            pixel_values=pv, image_grid_thw=thw, mm_token_type_ids=mm,
        )
        return gather_log_probs(logits, input_ids)

    def step(self, samples: List[GateSampleVL], verbose: bool = False) -> Dict[str, float]:
        if not samples:
            return {"n_samples": 0}
        toks, keep = [], []
        for s in samples:
            t = self._tokenize(s)
            if t is not None:
                toks.append(t); keep.append(s)
        if not toks:
            return {"n_samples": 0, "note": "no_gate_tokens"}
        samples = keep

        self.model.eval()
        old_lps = []
        with torch.no_grad():
            for t in toks:
                old_lps.append(self._forward(t).detach())

        self.model.train()
        agg = {"gate_loss": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0}
        stop = False
        for _epoch in range(self.ppo_epochs):
            if stop:
                break
            for i, (s, t) in enumerate(zip(samples, toks)):
                mask = t["gate_mask"].to(self.device)
                adv = torch.full(mask.shape, float(s.gate_advantage), device=self.device)
                log_probs_new = self._forward(t)
                loss, m = ppo_clip_loss(
                    log_prob_new=log_probs_new, log_prob_old=old_lps[i],
                    advantages=adv, mask=mask, ppo_clip=self.ppo_clip,
                )
                if m["approx_kl"] > self.target_kl:
                    stop = True
                    break
                if self.entropy_coeff > 0:
                    denom = mask.float().sum().clamp_min(1.0)
                    ent = (-log_probs_new * mask.float()).sum() / denom
                    loss = loss - self.entropy_coeff * ent
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                agg["gate_loss"] += float(loss.detach())
                agg["approx_kl"] += m["approx_kl"]
                agg["clip_frac"] += m["clip_frac"]
                agg["n_updates"] += 1
        n = max(1, agg["n_updates"])
        for k in ("gate_loss", "approx_kl", "clip_frac"):
            agg[k] /= n
        agg["n_samples"] = len(samples)
        return agg
