"""
cf_content_trainer_vl.py — VL (Option A / A2) content-side PPO trainer.

Forked from cf_content_trainer.py so the ALFWorld text trainer stays UNTOUCHED.
Differences vs the text trainer:
  * tokenization uses the model's AutoProcessor (Qwen3VLProcessor): the companion
    prompt is rebuilt as a multimodal message (frame image(s) + text), producing
    input_ids WITH expanded image tokens + pixel_values + image_grid_thw.
  * forward passes pixel_values / image_grid_thw to CompanionWithValueHead.
  * processed one sample at a time (mini_batch=1) to avoid VL pixel_values
    batching complexity — correctness first; batch later if needed.

Reuses ppo_clip_loss / gather_log_probs from cf_content_trainer (import only,
no modification). The advice-token mask reuses cf_token_masks.

Frames come from the HELP-state obs: sample.frames_b64 (base64 JPEG list),
originally from MCEnv.get_current_obs()["frames_b64"] carried in MainStep.state_dict.

STATUS: written without a VL box to test on; validate on the server
(Stage 4 smoke). Marked TODO where the Qwen3-VL processor API needs a live check.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from rl_causal.ppo.cf_token_masks import build_token_mask, TokenMask
from rl_causal.ppo.cf_content_trainer import ppo_clip_loss, gather_log_probs


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

@dataclass
class ContentSampleVL:
    """One (state, branch) VL content-training sample.

    Same as the text ContentSample plus `frames_b64` (the HELP-state view the
    companion saw). `system_prompt` is the companion system prompt used at rollout.
    """
    prompt_text: str                 # text part of the companion user prompt
    response_text: str               # raw JSON emitted for this branch's advice
    advantage: float
    frames_b64: List[str] = field(default_factory=list)
    system_prompt: str = ""
    q_target: Optional[float] = None
    state_prompt_text: Optional[str] = None
    state_frames_b64: Optional[List[str]] = None
    state_id: int = -1
    branch_id: int = -1
    branch_type: str = ""


def _decode_frames(frames_b64: List[str]) -> list:
    imgs = []
    from PIL import Image
    for b in frames_b64 or []:
        try:
            imgs.append(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB"))
        except Exception:
            pass
    return imgs


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ContentTrainerVL:
    """PPO clip trainer for advice tokens with a VL backbone. Mini-batch=1."""

    def __init__(
        self,
        model,                      # CompanionWithValueHead (vision=True, has .processor)
        lr_lora: float = 1e-5,
        lr_value: float = 5e-5,
        ppo_clip: float = 0.2,
        value_coeff: float = 0.5,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 2,
        max_seq_length: int = 4096,
        device: Optional[torch.device] = None,
    ) -> None:
        assert getattr(model, "processor", None) is not None, \
            "ContentTrainerVL needs a VL model loaded with vision=True (has .processor)"
        self.model = model
        self.processor = model.processor
        self.tok = model.processor.tokenizer
        self.ppo_clip = ppo_clip
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.max_seq_length = max_seq_length
        self.device = device or next(model.parameters()).device

        pg = model.trainable_params_by_group()
        self.optimizer = torch.optim.AdamW([
            {"params": pg["lora"], "lr": lr_lora},
            {"params": pg["value_head"], "lr": lr_value},
        ])

    # ---- tokenization (multimodal) --------------------------------------

    def _tokenize(self, sample: ContentSampleVL) -> Optional[Dict[str, Any]]:
        """Build model inputs for one sample: multimodal prompt + text response.

        Returns dict with input_ids (1,L), attention_mask (1,L), pixel_values,
        image_grid_thw, advice_mask (1,L bool), prompt_len. None if no advice tokens.
        """
        imgs = _decode_frames(sample.frames_b64)
        # Multimodal user content: image placeholders + text. If no frames, this
        # degrades to a text-only prompt (still valid).
        user_content = [{"type": "image"} for _ in imgs]
        user_content.append({"type": "text", "text": sample.prompt_text})
        messages = [
            {"role": "system", "content": sample.system_prompt or ""},
            {"role": "user", "content": user_content},
        ]
        # TODO(server): confirm Qwen3VLProcessor supports apply_chat_template like this.
        prompt_str = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        proc = self.processor(
            text=[prompt_str],
            images=imgs if imgs else None,
            return_tensors="pt",
        )
        prompt_ids = proc["input_ids"][0]                       # (Lp,)
        response_ids = self.tok(sample.response_text, add_special_tokens=False)["input_ids"]
        prompt_len = int(prompt_ids.shape[0])

        full_ids = torch.cat(
            [prompt_ids, torch.tensor(response_ids, dtype=prompt_ids.dtype)], dim=0
        )
        if full_ids.shape[0] > self.max_seq_length:
            # keep the response; drop from the FRONT of the prompt is unsafe with
            # image tokens, so we simply skip over-long samples.
            return None

        # Advice mask over the response region only (image tokens live in prompt).
        mask = torch.zeros(full_ids.shape[0], dtype=torch.bool)
        tm: TokenMask = build_token_mask(self.tok, sample.response_text, response_ids)
        for local_idx in tm.advice_positions:
            fi = prompt_len + local_idx
            if 0 <= fi < mask.shape[0]:
                mask[fi] = True
        if not bool(mask.any()):
            return None

        # mm_token_type_ids: the processor returns it for the prompt (marks image
        # tokens for M-RoPE). Extend with 0 (text) for the appended response tokens.
        mm_full = None
        mm_prompt = proc.get("mm_token_type_ids")
        if mm_prompt is not None:
            pm = mm_prompt[0]
            mm_full = torch.cat(
                [pm, torch.zeros(len(response_ids), dtype=pm.dtype)], dim=0
            ).unsqueeze(0)

        out = {
            "input_ids": full_ids.unsqueeze(0),
            "attention_mask": torch.ones(1, full_ids.shape[0], dtype=torch.long),
            "advice_mask": mask.unsqueeze(0),
            "prompt_len": prompt_len,
            "pixel_values": proc.get("pixel_values"),
            "image_grid_thw": proc.get("image_grid_thw"),
            "mm_token_type_ids": mm_full,
        }
        return out

    def _forward(self, tok: Dict[str, Any], need_value: bool):
        input_ids = tok["input_ids"].to(self.device)
        attn = tok["attention_mask"].to(self.device)
        pv = tok["pixel_values"].to(self.device) if tok.get("pixel_values") is not None else None
        thw = tok["image_grid_thw"].to(self.device) if tok.get("image_grid_thw") is not None else None
        mm = tok["mm_token_type_ids"].to(self.device) if tok.get("mm_token_type_ids") is not None else None
        logits, value = self.model(
            input_ids=input_ids, attention_mask=attn, return_logits=True,
            pixel_values=pv, image_grid_thw=thw, mm_token_type_ids=mm,
        )
        log_probs = gather_log_probs(logits, input_ids)         # (1, L)
        return log_probs, value

    # ---- update ---------------------------------------------------------

    def step(self, samples: List[ContentSampleVL], verbose: bool = False) -> Dict[str, float]:
        if not samples:
            return {"n_samples": 0}
        toks = []
        keep = []
        for s in samples:
            t = self._tokenize(s)
            if t is not None:
                toks.append(t); keep.append(s)
        if not toks:
            return {"n_samples": 0, "note": "no_advice_tokens"}
        samples = keep

        # cache old log-probs (per sample, no grad)
        self.model.eval()
        old_lps = []
        with torch.no_grad():
            for t in toks:
                lp, _ = self._forward(t, need_value=False)
                old_lps.append(lp.detach())

        self.model.train()
        agg = {"policy_loss": 0.0, "value_loss": 0.0, "approx_kl": 0.0,
               "clip_frac": 0.0, "n_updates": 0}
        for _epoch in range(self.ppo_epochs):
            for i, (s, t) in enumerate(zip(samples, toks)):
                mask = t["advice_mask"].to(self.device)
                adv = torch.full(mask.shape, float(s.advantage), device=self.device)
                need_value = s.q_target is not None
                log_probs_new, value = self._forward(t, need_value=need_value)
                policy_loss, m = ppo_clip_loss(
                    log_prob_new=log_probs_new, log_prob_old=old_lps[i],
                    advantages=adv, mask=mask, ppo_clip=self.ppo_clip,
                )
                value_loss = torch.tensor(0.0, device=policy_loss.device)
                if need_value and value is not None:
                    q = torch.tensor([float(s.q_target)], dtype=torch.float32,
                                     device=value.device)
                    value_loss = F.mse_loss(value, q)
                loss = policy_loss + self.value_coeff * value_loss.to(policy_loss.device)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    self.max_grad_norm,
                )
                self.optimizer.step()

                agg["policy_loss"] += float(policy_loss.detach())
                agg["value_loss"] += float(value_loss.detach())
                agg["approx_kl"] += m["approx_kl"]
                agg["clip_frac"] += m["clip_frac"]
                agg["n_updates"] += 1
        n = max(1, agg["n_updates"])
        for k in ("policy_loss", "value_loss", "approx_kl", "clip_frac"):
            agg[k] /= n
        agg["n_samples"] = len(samples)
        return agg
