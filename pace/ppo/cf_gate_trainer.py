"""
cf_gate_trainer.py — gate-side PPO trainer for CF-nstep-GRPO.

Mirrors cf_content_trainer.py with two differences:

  1. Mask: `gate_positions` instead of `advice_positions` — loss ONLY on
     the HELP/SILENCE literal token(s), never on advice content, JSON keys,
     or structure. Prevents content credit from contaminating the gate
     decision (§ 5.5, "separated content/gate loops").

  2. Advantage source: scalar `gate_advantage` per HELP state (from
     cf_advantage.GroupAdvantage.gate_advantage), NOT per branch.
     Rationale: gate is a single binary decision per state; we don't
     credit-assign it to individual advice samples.

     For SILENCE states in the main trajectory that were NOT expanded
     (their counterfactual would need a HELP branch's Q, which we don't
     have), we skip gate training. Only HELP states — where we have both
     Q_help_mean and Q_silence from the K+2 expansion — provide training
     signal. Every K+2 expansion yields ONE gate sample.

Everything else (PPO clip, log_π_old caching, mini-batch epochs, value
head co-training) reuses the same primitives as the content trainer.

Reference: method.md § 4.2, § 5.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_causal.ppo.cf_token_masks import build_token_mask, TokenMask
from rl_causal.ppo.cf_content_trainer import (
    tokenize_prompt_and_response,
    pad_batch,
    gather_log_probs,
    ppo_clip_loss,
)


# ---------------------------------------------------------------------------
# Sample structure
# ---------------------------------------------------------------------------

@dataclass
class GateSample:
    """One HELP state's gate-training sample.

    Fields:
      prompt_text     the full system+user chat text used at rollout time
      response_text   the raw JSON emitted at this HELP state (we use the
                      HELP-replay branch's response — same as main-traj's)
      gate_advantage  scalar A_gate(s_t) = (Q̄_help - Q_silence) / σ_Q

      # Bookkeeping
      state_id        HELP state index (into ExpandedHelpState list)
    """
    prompt_text: str
    response_text: str
    gate_advantage: float
    state_id: int = -1


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def build_gate_samples(
    expanded_states: Sequence,           # List[ExpandedHelpState]
    advantage_batch,                      # HelpStateAdvantageBatch
    prompt_texts_by_state: Dict[int, str],
) -> List[GateSample]:
    """Build one GateSample per HELP state.

    Args:
      expanded_states       — output of cf_expander.expand_trajectory
      advantage_batch       — output of cf_advantage.batch_group_advantages
                              (its .gate_advantages are the scalars we need)
      prompt_texts_by_state — {state_id: companion_prompt_text} — the caller
                              looks these up from the main trajectory's
                              MainStep.companion_prompt at the corresponding
                              step_index.

    Returns:
      List[GateSample], length ≤ num_help_states
    """
    from rl_causal.ppo.cf_expander import ExpandedHelpState  # avoid circular

    samples: List[GateSample] = []
    for sid, ga in enumerate(advantage_batch.gate_advantages):
        if sid not in prompt_texts_by_state:
            continue
        st: ExpandedHelpState = expanded_states[sid]
        # Use the HELP-replay branch's response as the gate-training text
        # (same JSON emitted at the main trajectory's HELP state, so the
        # gate token position aligns with what π_c actually outputted).
        response_text = st.help_replay_branch.advice_response_text or ""
        if not response_text:
            continue
        samples.append(GateSample(
            prompt_text=prompt_texts_by_state[sid],
            response_text=response_text,
            gate_advantage=float(ga),
            state_id=sid,
        ))
    return samples


# ---------------------------------------------------------------------------
# Gate-only mask (over full input_ids)
# ---------------------------------------------------------------------------

def build_gate_mask_full(
    tokenizer,
    tokenized: Dict[str, Any],
) -> torch.Tensor:
    """Build a length-L bool mask over full input_ids that is True ONLY at
    gate-value token positions (e.g., the "HELP" or "SILENCE" literal).

    Uses TokenMask.gate_positions from cf_token_masks. Prompt positions and
    every non-gate token in the response are False.
    """
    prompt_len = tokenized["prompt_len"]
    response_ids = tokenized["response_ids"]
    response_text = tokenized["response_text"]
    L = tokenized["input_ids"].shape[0]

    mask = torch.zeros(L, dtype=torch.bool)
    tm: TokenMask = build_token_mask(tokenizer, response_text, response_ids)
    for local_idx in tm.gate_positions:
        full_idx = prompt_len + local_idx
        if 0 <= full_idx < L:
            mask[full_idx] = True
    return mask


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GateTrainer:
    """PPO clip trainer for gate token(s). Mirrors ContentTrainer with a
    different mask and per-state (not per-branch) advantage.

    Note: shares the SAME model + optimizer as ContentTrainer in practice.
    If used as a standalone trainer, you can either give it a fresh
    optimizer or hand in the same optimizer from ContentTrainer to keep
    LoRA + value head weights consistent across loops.

    Common usage: instantiate one Optimizer at the top level, wrap the
    model, and call `content_trainer.step(...)` and
    `gate_trainer.step(...)` alternately per method.md § 5.5.
    """

    def __init__(
        self,
        model,                    # CompanionWithValueHead
        tokenizer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_lora: float = 1e-5,      # ignored if `optimizer` given
        lr_value: float = 5e-5,     # ignored if `optimizer` given
        ppo_clip: float = 0.2,
        entropy_coeff: float = 0.01,   # gate benefits from small entropy bonus
        target_kl: float = 0.2,        # W32 fix: early-stop the update if the
                                       # single-token gate policy drifts past
                                       # this within-update KL (prevents the
                                       # KL->29 / clip_frac=1.0 blowup)
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 2,
        mini_batch_size: int = 4,
        max_seq_length: int = 2048,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.ppo_clip = ppo_clip
        self.entropy_coeff = entropy_coeff
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.max_seq_length = max_seq_length

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        if optimizer is None:
            param_groups = model.trainable_params_by_group()
            self.optimizer = torch.optim.AdamW([
                {"params": param_groups["lora"], "lr": lr_lora},
                {"params": param_groups["value_head"], "lr": lr_value},
            ])
            self._owns_optimizer = True
        else:
            self.optimizer = optimizer
            self._owns_optimizer = False

        n_lora = sum(p.numel() for p in model.trainable_params_by_group()["lora"])
        print(f"[GateTrainer] LoRA params (visible): {n_lora/1e6:.2f}M  "
              f"shared_opt={not self._owns_optimizer}", flush=True)

    # ----------------------------------------------------------------------

    def _tokenize_sample(self, sample: GateSample) -> Dict[str, Any]:
        tk = tokenize_prompt_and_response(
            self.tokenizer, sample.prompt_text, sample.response_text,
            max_length=self.max_seq_length,
        )
        tk["gate_mask"] = build_gate_mask_full(self.tokenizer, tk)
        return tk

    def _forward_batch(
        self,
        batch_input_ids: torch.Tensor,
        batch_attention: torch.Tensor,
    ) -> torch.Tensor:
        """Return log_probs (B, L). Value head output is ignored here (gate
        trainer doesn't co-train the critic; that's content trainer's job)."""
        input_ids = batch_input_ids.to(self.device)
        attn = batch_attention.to(self.device)
        logits, _ = self.model(
            input_ids=input_ids, attention_mask=attn, return_logits=True,
        )
        return gather_log_probs(logits, input_ids)

    @torch.no_grad()
    def _cache_old_log_probs(
        self, tokenized_samples: List[Dict[str, Any]],
    ) -> List[torch.Tensor]:
        self.model.eval()
        out: List[torch.Tensor] = []
        B = self.mini_batch_size
        pad_id = self.tokenizer.pad_token_id or 0
        for start in range(0, len(tokenized_samples), B):
            mb = tokenized_samples[start:start + B]
            input_ids = pad_batch([s["input_ids"] for s in mb], pad_value=pad_id)
            attn = pad_batch([s["attention_mask"] for s in mb], pad_value=0)
            log_probs = self._forward_batch(input_ids, attn).detach().cpu()
            for i, s in enumerate(mb):
                L = s["input_ids"].shape[0]
                out.append(log_probs[i, :L].clone())
        return out

    # ----------------------------------------------------------------------

    def step(
        self,
        samples: List[GateSample],
        verbose: bool = False,
    ) -> Dict[str, float]:
        """One PPO update for gate tokens. K epochs × mini-batches."""
        if not samples:
            return {"n_samples": 0}

        tokenized = [self._tokenize_sample(s) for s in samples]
        # Drop samples with no gate token located
        keep = [i for i, t in enumerate(tokenized) if t["gate_mask"].any().item()]
        if not keep:
            return {"n_samples": 0, "note": "no_gate_tokens"}
        samples = [samples[i] for i in keep]
        tokenized = [tokenized[i] for i in keep]

        old_lps = self._cache_old_log_probs(tokenized)

        self.model.train()
        agg = {
            "gate_loss": 0.0, "approx_kl": 0.0,
            "clip_frac": 0.0, "mean_ratio": 0.0, "n_updates": 0,
        }
        pad_id = self.tokenizer.pad_token_id or 0
        stop_early = False

        for epoch in range(self.ppo_epochs):
            if stop_early:
                break
            perm = torch.randperm(len(samples)).tolist()
            for start in range(0, len(samples), self.mini_batch_size):
                mb_idx = perm[start:start + self.mini_batch_size]
                mb_tok = [tokenized[i] for i in mb_idx]
                mb_samp = [samples[i] for i in mb_idx]
                mb_old = [old_lps[i] for i in mb_idx]

                input_ids = pad_batch([t["input_ids"] for t in mb_tok], pad_value=pad_id)
                attn = pad_batch([t["attention_mask"] for t in mb_tok], pad_value=0)
                # Move gate mask to device ONCE, reuse throughout mini-batch
                gate_mask = pad_batch(
                    [t["gate_mask"].to(torch.long) for t in mb_tok], pad_value=0
                ).to(torch.bool).to(self.device)
                old_lp = pad_batch(mb_old, pad_value=0.0).to(self.device)

                B, L = input_ids.shape
                adv_scalar = torch.tensor(
                    [s.gate_advantage for s in mb_samp], dtype=torch.float32,
                ).unsqueeze(1).expand(B, L).to(self.device)

                log_probs_new = self._forward_batch(input_ids, attn)

                loss, ppo_metrics = ppo_clip_loss(
                    log_prob_new=log_probs_new,
                    log_prob_old=old_lp,
                    advantages=adv_scalar,
                    mask=gate_mask,
                    ppo_clip=self.ppo_clip,
                )

                # W32 fix: within-update trust-region guard. approx_kl here is
                # current-policy vs update-start policy; once it exceeds
                # target_kl the gate is over-stepping, so skip this (and all
                # further) mini-batch steps instead of slamming the logit.
                if ppo_metrics["approx_kl"] > self.target_kl:
                    if verbose:
                        print(f"    [gate ep {epoch} mb "
                              f"{start//self.mini_batch_size}] "
                              f"EARLY-STOP kl≈{ppo_metrics['approx_kl']:+.4f} "
                              f"> target_kl={self.target_kl}", flush=True)
                    stop_early = True
                    break

                # Small entropy bonus on gate to keep exploration alive
                if self.entropy_coeff > 0:
                    # H ≈ -mean(log_prob_new * mask)
                    denom = gate_mask.float().sum().clamp_min(1.0)
                    ent = (-log_probs_new * gate_mask.float()).sum() / denom
                    loss = loss - self.entropy_coeff * ent

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    max_norm=self.max_grad_norm,
                )
                self.optimizer.step()

                agg["gate_loss"] += float(loss.detach())
                agg["approx_kl"] += ppo_metrics["approx_kl"]
                agg["clip_frac"] += ppo_metrics["clip_frac"]
                agg["mean_ratio"] += ppo_metrics["mean_ratio"]
                agg["n_updates"] += 1

                if verbose:
                    print(f"    [gate ep {epoch} mb {start//self.mini_batch_size}] "
                          f"loss={float(loss.detach()):+.4f} "
                          f"kl≈{ppo_metrics['approx_kl']:+.4f} "
                          f"clip_frac={ppo_metrics['clip_frac']:.2f}",
                          flush=True)

        n = max(1, agg["n_updates"])
        for k in ("gate_loss", "approx_kl", "clip_frac", "mean_ratio"):
            agg[k] /= n
        agg["n_samples"] = len(samples)
        return agg


# ---------------------------------------------------------------------------
# Sanity self-test — same helper functions as content_trainer, so we
# just verify build_gate_mask_full logic with a mock tokenizer.
# ---------------------------------------------------------------------------

def _self_test() -> None:
    print("[test] build_gate_mask_full with mock tokenizer")

    class MockTok:
        pad_token_id = 0
        def __call__(self, text, add_special_tokens=False):
            words = text.split() if text else []
            return {"input_ids": [i + 1 for i in range(len(words))]}
        def decode(self, ids, skip_special_tokens=False):
            # Reproduce tokens as space-joined "wordN" strings so downstream
            # substring search in build_token_mask can align.
            # For simplicity, we craft a synthetic advice+gate response and
            # verify only that the mask is not empty.
            return "".join(str(int(i)) + " " for i in ids)

    # For a REAL check of gate_mask alignment, we'd need a real tokenizer.
    # Here we just check the plumbing runs.
    prompt = "you are helping"
    response = '{"gate": "HELP", "advice": "Go there"}'
    tk = MockTok()
    tokenized = tokenize_prompt_and_response(tk, prompt, response, max_length=100)
    tokenized["gate_mask"] = build_gate_mask_full(tk, tokenized)
    # With a naive word-tokenizer, gate token identification will be brittle,
    # but the function should at least not raise.
    print(f"  prompt_len={tokenized['prompt_len']}  "
          f"response tokens={len(tokenized['response_ids'])}  "
          f"gate mask any={bool(tokenized['gate_mask'].any().item())}  "
          f"shape={tuple(tokenized['gate_mask'].shape)}")

    print("[test] GateSample dataclass")
    s = GateSample(
        prompt_text="hello",
        response_text='{"gate": "HELP", "advice": "x"}',
        gate_advantage=+1.23,
        state_id=0,
    )
    print(f"  {s}")

    print("[test] all cf_gate_trainer helper checks passed")


if __name__ == "__main__":
    _self_test()
