"""
cf_content_trainer.py — content-side PPO trainer for CF-nstep-GRPO.

Trains the companion's advice content (the string inside {"advice": "..."})
using a GRPO-style group-relative advantage combined with a PPO clip loss.

Per method.md § 4.2, § 5.5:

  loss_content = - mean_over_advice_tokens[
                     min( r_t * Â_i, clip(r_t, 1-ε, 1+ε) * Â_i )
                 ]
  + value_coeff * MSE( V(s), Q_target )
  + entropy_coeff * H(π)       (optional, off by default)

where:
  r_t = exp( log π_new(y_t | s, y_<t) - log π_old(y_t | s, y_<t) )
  Â_i = the K+1 group-relative advantage from cf_advantage
        (broadcast across all advice tokens of branch i)

Key design points:
  1. **Masking**: PPO loss ONLY on advice-value tokens (via cf_token_masks).
     Prompt, JSON keys, whitespace, gate value — all masked out. Ensures
     content credit doesn't leak into gate/format positions.
  2. **log π_old caching**: cache once at the start of each update, replay
     across K PPO epochs. Standard PPO recipe.
  3. **Value head co-training**: uses n-step Q targets from cf_nstep_q on
     state-only prompts. Kept in the same optimizer step for efficiency.
  4. **No KL-to-ref penalty in MVP**: PPO clip already bounds drift. If
     training becomes unstable, add via `kl_ref_coeff` (currently ignored).

Reference:
  - method.md § 4.2 (advantage), § 5.5 (training loop)
  - md/SPARK_finetune_unified.md § v4 refinements
  - DeepSeek GRPO paper (group-relative baseline shape)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_causal.ppo.cf_token_masks import build_token_mask, TokenMask


# ---------------------------------------------------------------------------
# Sample structure
# ---------------------------------------------------------------------------

@dataclass
class ContentSample:
    """One (state, branch) training sample for the content PPO loss.

    Fields:
      prompt_text      the full system+user chat text used at rollout time
                       (companion_prompt from MainStep / expander)
      response_text    the raw JSON emitted by the companion sample that
                       produced this branch's advice
      advantage        Â_i scalar from cf_advantage (already shape-corrected
                       with intrinsic + diversity terms baked in)

      # Q-target for value head (optional; not all samples have one)
      q_target         n-step Q at the state prompt, or None if not paired
      state_prompt     state-only prompt for V(s) forward, or None

      # Bookkeeping (for logging / debugging)
      state_id         which HELP state this sample came from
      branch_id        which of the K+2 branches within that state
      branch_type      "silence" | "help_replay" | "advice"
    """
    prompt_text: str
    response_text: str
    advantage: float

    q_target: Optional[float] = None
    state_prompt: Optional[str] = None

    state_id: int = -1
    branch_id: int = -1
    branch_type: str = ""


# ---------------------------------------------------------------------------
# Sample construction from ExpandedHelpState + HelpStateAdvantageBatch
# ---------------------------------------------------------------------------

def build_content_samples(
    expanded_states: Sequence,          # List[ExpandedHelpState]
    advantage_batch,                     # HelpStateAdvantageBatch
    q_targets_by_state: Optional[Dict[int, Tuple[str, float]]] = None,
    skip_silence: bool = True,
) -> List[ContentSample]:
    """Assemble content-training samples from CF expansion + advantage batch.

    Args:
      expanded_states     — output of cf_expander.expand_trajectory
      advantage_batch     — output of cf_advantage.batch_group_advantages
      q_targets_by_state  — optional {state_id: (state_prompt, q_target)}
                            for value-head co-training. Usually built from
                            n-step Q along the main trajectory.
      skip_silence        — True: don't train content on SILENCE branch
                            (its advice is empty; nothing to learn).
                            False: include with empty advice (advice_positions
                            will be empty; loss contribution is 0 anyway).

    Returns:
      List[ContentSample], length ≤ num_help_states * (K+2)
    """
    from rl_causal.ppo.cf_expander import ExpandedHelpState  # avoid circular

    samples: List[ContentSample] = []
    for flat_i, sid in enumerate(advantage_batch.flat_state_ids):
        bid = advantage_batch.flat_branch_ids[flat_i]
        adv = advantage_batch.flat_advantages[flat_i]

        st: ExpandedHelpState = expanded_states[sid]

        # Which branch does bid refer to? Convention in cf_advantage:
        #   bid 0 = silence, bid 1 = help_replay, bid 2..K+1 = advice_branches[0..K-1]
        if bid == 0:
            branch = st.silence_branch
            branch_type = "silence"
        elif bid == 1:
            branch = st.help_replay_branch
            branch_type = "help_replay"
        else:
            branch = st.advice_branches[bid - 2]
            branch_type = "advice"

        if skip_silence and branch_type == "silence":
            continue

        # Build sample; need the prompt text from the main trajectory step
        # Access to companion_prompt: we need to look it up. ExpandedHelpState
        # doesn't carry it directly; caller should provide it OR we resolve
        # it via main_trajectory. For now, assume expander samples put
        # response_text = advice_response_text (which they do).
        response_text = branch.advice_response_text or ""
        # If a caller wants zero-length silence samples included with a
        # placeholder response, they can post-hoc set it. We tolerate empty.
        if not response_text:
            continue

        # Q target for the state (shared across branches at same state — but
        # only attach it to ONE branch to avoid duplicate value updates).
        q_target = None
        state_prompt = None
        if q_targets_by_state is not None and bid == 1:
            # Attach Q target to help_replay branch (arbitrary but consistent)
            if sid in q_targets_by_state:
                state_prompt, q_target = q_targets_by_state[sid]

        samples.append(ContentSample(
            prompt_text="",                          # filled by caller
            response_text=response_text,
            advantage=float(adv),
            q_target=q_target,
            state_prompt=state_prompt,
            state_id=sid,
            branch_id=bid,
            branch_type=branch_type,
        ))
    return samples


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def tokenize_prompt_and_response(
    tokenizer,
    prompt_text: str,
    response_text: str,
    max_length: int = 2048,
) -> Dict[str, Any]:
    """Tokenize prompt + response separately, then concat.

    Returns dict with:
      input_ids      long tensor (L,) — prompt_ids + response_ids
      attention_mask long tensor (L,) — all 1s (no padding at sample level)
      prompt_len     int — number of prompt tokens
      response_ids   list[int] — response tokens only, for TokenMask building
      response_text  str
    """
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    full = prompt_ids + response_ids
    if len(full) > max_length:
        # Truncate from prompt side (keep response intact for gradient targets)
        overflow = len(full) - max_length
        prompt_ids = prompt_ids[overflow:]
        full = prompt_ids + response_ids
    input_ids = torch.tensor(full, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_len": len(prompt_ids),
        "response_ids": response_ids,
        "response_text": response_text,
    }


def build_advice_mask_full(
    tokenizer,
    tokenized: Dict[str, Any],
) -> torch.Tensor:
    """Build a length-L bool mask over full input_ids that is True ONLY at
    advice-value token positions.

    NOTE: The mask is at "generation" positions, i.e., position `t` means
    "the token predicted at step t from logits[t-1]". So mask index 0 is
    never a training target (nothing predicts it). All prompt positions are
    False; advice tokens within response are True; other response tokens
    (JSON keys, gate value, structure) are False.
    """
    prompt_len = tokenized["prompt_len"]
    response_ids = tokenized["response_ids"]
    response_text = tokenized["response_text"]
    L = tokenized["input_ids"].shape[0]

    mask = torch.zeros(L, dtype=torch.bool)
    # Locate advice tokens within response
    tm: TokenMask = build_token_mask(tokenizer, response_text, response_ids)
    for local_idx in tm.advice_positions:
        full_idx = prompt_len + local_idx
        if 0 <= full_idx < L:
            mask[full_idx] = True
    return mask


def build_full_output_mask_full(
    tokenizer,
    tokenized: Dict[str, Any],
) -> torch.Tensor:
    """Mask True at BOTH gate-value and advice-value tokens (the whole decision).

    Used by ablation 3c (GRPO): the advantage is applied to the full {gate,advice}
    output rather than to advice tokens only, so gate and advice are trained
    jointly by one PPO pass (no separate gate credit)."""
    prompt_len = tokenized["prompt_len"]
    response_ids = tokenized["response_ids"]
    response_text = tokenized["response_text"]
    L = tokenized["input_ids"].shape[0]

    mask = torch.zeros(L, dtype=torch.bool)
    tm: TokenMask = build_token_mask(tokenizer, response_text, response_ids)
    for local_idx in list(tm.gate_positions) + list(tm.advice_positions):
        full_idx = prompt_len + local_idx
        if 0 <= full_idx < L:
            mask[full_idx] = True
    return mask


def pad_batch(
    tensors: List[torch.Tensor],
    pad_value: int = 0,
) -> torch.Tensor:
    """Right-pad a list of 1-D tensors to the same length."""
    max_len = max(t.shape[0] for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, :t.shape[0]] = t
    return out


# ---------------------------------------------------------------------------
# Log-prob extraction from logits
# ---------------------------------------------------------------------------

def gather_log_probs(
    logits: torch.Tensor,      # (B, L, V)
    labels: torch.Tensor,      # (B, L) target token id at each position
) -> torch.Tensor:
    """log π(y_t | y_<t) at each position. Position t uses logits[:, t-1, :]
    to predict labels[:, t]. Position 0 has no prediction; log_prob at
    position 0 is set to 0 (caller must mask it out).

    Returns:
      log_probs (B, L) with log_probs[:, 0] = 0.
    """
    # Shift logits left so logits_shifted[:, t] predicts labels[:, t+1]
    logits_shifted = logits[:, :-1, :]                        # (B, L-1, V)
    labels_shifted = labels[:, 1:]                            # (B, L-1)
    log_probs_all = F.log_softmax(logits_shifted, dim=-1)     # (B, L-1, V)
    # Gather at label positions
    gathered = log_probs_all.gather(
        2, labels_shifted.unsqueeze(-1)
    ).squeeze(-1)                                             # (B, L-1)
    # Prepend a zero for position 0 (no prediction available)
    B = logits.shape[0]
    zero_col = torch.zeros(B, 1, device=logits.device, dtype=gathered.dtype)
    out = torch.cat([zero_col, gathered], dim=1)              # (B, L)
    return out


# ---------------------------------------------------------------------------
# PPO clip loss
# ---------------------------------------------------------------------------

def ppo_clip_loss(
    log_prob_new: torch.Tensor,   # (B, L)
    log_prob_old: torch.Tensor,   # (B, L)
    advantages: torch.Tensor,      # (B, L) — broadcast per-token from scalar
    mask: torch.Tensor,            # (B, L) bool
    ppo_clip: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Standard PPO clip loss, masked to `mask == True`.

    Returns:
      loss (scalar) — negated so we minimize
      metrics dict — approx KL, clip fraction, mean ratio
    """
    log_ratio = log_prob_new - log_prob_old.detach()
    # Numerical guard: clamp log_ratio to avoid overflow in exp
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages
    per_token_loss = -torch.min(surr1, surr2)                # negate for min

    denom = mask.float().sum().clamp_min(1.0)
    loss = (per_token_loss * mask.float()).sum() / denom

    with torch.no_grad():
        # k3 estimator (Schulman blog, always >= 0):
        # KL ≈ E[ (π_old/π_new) - 1 - log(π_old/π_new) ]
        #    = E[ exp(-log_ratio) - 1 + log_ratio ]
        neg_log_ratio = -log_ratio
        k3 = (torch.exp(neg_log_ratio) - 1.0 + log_ratio)
        approx_kl = (k3 * mask.float()).sum() / denom
        clip_frac = (((ratio - 1.0).abs() > ppo_clip).float() * mask.float()).sum() / denom
        mean_ratio = (ratio * mask.float()).sum() / denom
    return loss, {
        "approx_kl": float(approx_kl.detach()),
        "clip_frac": float(clip_frac.detach()),
        "mean_ratio": float(mean_ratio.detach()),
    }


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------

class ContentTrainer:
    """PPO clip trainer for advice content tokens.

    Usage:
        model = CompanionWithValueHead.from_pretrained(...)
        trainer = ContentTrainer(model, tokenizer, lr_lora=1e-5, lr_value=5e-5)
        for update in range(N):
            samples = build_content_samples(expanded_states, adv_batch, q_targets)
            metrics = trainer.step(samples)
    """

    def __init__(
        self,
        model,                    # CompanionWithValueHead
        tokenizer,
        lr_lora: float = 1e-5,
        lr_value: float = 5e-5,
        ppo_clip: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.0,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 2,
        mini_batch_size: int = 4,
        max_seq_length: int = 2048,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.ppo_clip = ppo_clip
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.max_seq_length = max_seq_length

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        # Set up optimizer with per-group LR (LoRA vs value head)
        param_groups = model.trainable_params_by_group()
        self.optimizer = torch.optim.AdamW([
            {"params": param_groups["lora"], "lr": lr_lora},
            {"params": param_groups["value_head"], "lr": lr_value},
        ])
        n_lora = sum(p.numel() for p in param_groups["lora"])
        n_val = sum(p.numel() for p in param_groups["value_head"])
        print(f"[ContentTrainer] LoRA params: {n_lora/1e6:.2f}M  "
              f"value head: {n_val/1e6:.2f}M", flush=True)

    # ----------------------------------------------------------------------
    # Per-sample forward → log_probs + advice_mask
    # ----------------------------------------------------------------------

    def _tokenize_sample(self, sample: ContentSample) -> Dict[str, Any]:
        tk = tokenize_prompt_and_response(
            self.tokenizer, sample.prompt_text, sample.response_text,
            max_length=self.max_seq_length,
        )
        # ablation 3c (GRPO): train the full {gate,advice} output, not advice only
        if getattr(self, "full_output_mask", False):
            tk["advice_mask"] = build_full_output_mask_full(self.tokenizer, tk)
        else:
            tk["advice_mask"] = build_advice_mask_full(self.tokenizer, tk)
        return tk

    def _forward_batch(
        self,
        batch_input_ids: torch.Tensor,   # (B, L)
        batch_attention: torch.Tensor,   # (B, L)
        return_value: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward through CompanionWithValueHead.

        Returns:
          logits    (B, L, V)
          log_probs (B, L)   — computed via gather_log_probs
          values    (B,) or None
        """
        input_ids = batch_input_ids.to(self.device)
        attn = batch_attention.to(self.device)
        logits, values = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            return_logits=True,
        )
        log_probs = gather_log_probs(logits, input_ids)
        return logits, log_probs, (values if return_value else None)

    # ----------------------------------------------------------------------
    # Cache log_π_old
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def _cache_old_log_probs(
        self,
        tokenized_samples: List[Dict[str, Any]],
    ) -> List[torch.Tensor]:
        """One forward pass with current policy to record log_π_old per sample.

        Uses mini-batching to keep memory bounded.
        """
        self.model.eval()
        old_lps: List[torch.Tensor] = []
        B = self.mini_batch_size
        for start in range(0, len(tokenized_samples), B):
            mb = tokenized_samples[start:start + B]
            input_ids = pad_batch(
                [s["input_ids"] for s in mb],
                pad_value=self.tokenizer.pad_token_id or 0,
            )
            attn = pad_batch([s["attention_mask"] for s in mb], pad_value=0)
            _, log_probs, _ = self._forward_batch(input_ids, attn, return_value=False)
            log_probs = log_probs.detach().cpu()
            for i, s in enumerate(mb):
                L = s["input_ids"].shape[0]
                old_lps.append(log_probs[i, :L].clone())
        return old_lps

    # ----------------------------------------------------------------------
    # Update step
    # ----------------------------------------------------------------------

    def step(
        self,
        samples: List[ContentSample],
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Run one PPO update: K epochs × mini-batches over `samples`.

        Returns aggregated metrics dict.
        """
        if not samples:
            return {"n_samples": 0}

        # 1) Tokenize all samples (once)
        tokenized: List[Dict[str, Any]] = [self._tokenize_sample(s) for s in samples]
        # Filter out samples with zero advice tokens (would contribute 0 loss)
        keep_idx = [i for i, t in enumerate(tokenized) if t["advice_mask"].any().item()]
        if not keep_idx:
            return {"n_samples": 0, "note": "no_advice_tokens"}
        samples = [samples[i] for i in keep_idx]
        tokenized = [tokenized[i] for i in keep_idx]

        # 2) Cache log_π_old
        old_log_probs = self._cache_old_log_probs(tokenized)

        # 3) K PPO epochs
        self.model.train()
        agg = {
            "policy_loss": 0.0, "value_loss": 0.0,
            "approx_kl": 0.0, "clip_frac": 0.0, "mean_ratio": 0.0,
            "n_updates": 0,
        }
        pad_id = self.tokenizer.pad_token_id or 0

        for epoch in range(self.ppo_epochs):
            perm = torch.randperm(len(samples)).tolist()
            for start in range(0, len(samples), self.mini_batch_size):
                mb_idx = perm[start:start + self.mini_batch_size]
                mb_tok = [tokenized[i] for i in mb_idx]
                mb_samp = [samples[i] for i in mb_idx]
                mb_old = [old_log_probs[i] for i in mb_idx]

                # Pad + move to device ONCE per mini-batch
                input_ids = pad_batch([t["input_ids"] for t in mb_tok], pad_value=pad_id)
                attn = pad_batch([t["attention_mask"] for t in mb_tok], pad_value=0)
                advice_mask = pad_batch(
                    [t["advice_mask"].to(torch.long) for t in mb_tok], pad_value=0
                ).to(torch.bool).to(self.device)
                old_lp = pad_batch(mb_old, pad_value=0.0).to(self.device)

                # Per-token advantage: broadcast scalar per-sample across L
                B, L = input_ids.shape
                adv_scalar = torch.tensor(
                    [s.advantage for s in mb_samp], dtype=torch.float32,
                ).unsqueeze(1).expand(B, L).to(self.device)          # (B, L)

                # Forward
                needs_value = any(s.q_target is not None for s in mb_samp)
                _, log_probs_new, values_pred = self._forward_batch(
                    input_ids, attn, return_value=needs_value,
                )

                # PPO clip loss (advice mask only)
                policy_loss, ppo_metrics = ppo_clip_loss(
                    log_prob_new=log_probs_new,
                    log_prob_old=old_lp,
                    advantages=adv_scalar,
                    mask=advice_mask,
                    ppo_clip=self.ppo_clip,
                )

                # Value head loss on samples that have q_target
                value_loss = torch.tensor(0.0, device=self.device)
                if needs_value and values_pred is not None:
                    q_list, v_list = [], []
                    for i, s in enumerate(mb_samp):
                        if s.q_target is not None:
                            q_list.append(s.q_target)
                            v_list.append(values_pred[i])
                    if v_list:
                        v_tensor = torch.stack(v_list)
                        # q_tensor must match v_tensor's device (value head may
                        # be on a different GPU than main via device_map='auto')
                        q_tensor = torch.tensor(q_list, dtype=torch.float32,
                                                 device=v_tensor.device)
                        value_loss = F.mse_loss(v_tensor, q_tensor)

                # Ensure both losses live on the same device before adding
                # (value head can be on a different GPU under device_map='auto')
                loss = policy_loss + self.value_coeff * value_loss.to(policy_loss.device)

                # Optional entropy bonus (not used by default)
                if self.entropy_coeff > 0:
                    # Approx entropy at advice-token positions
                    # H ≈ -mean(log_prob_new) on advice mask
                    # advice_mask is already on device from above
                    ent = (-log_probs_new * advice_mask.float()).sum() / \
                          advice_mask.float().sum().clamp_min(1.0)
                    loss = loss - self.entropy_coeff * ent

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    max_norm=self.max_grad_norm,
                )
                self.optimizer.step()

                agg["policy_loss"] += float(policy_loss.detach())
                agg["value_loss"] += float(value_loss.detach())
                agg["approx_kl"] += ppo_metrics["approx_kl"]
                agg["clip_frac"] += ppo_metrics["clip_frac"]
                agg["mean_ratio"] += ppo_metrics["mean_ratio"]
                agg["n_updates"] += 1

                if verbose:
                    print(f"    [ep {epoch} mb {start//self.mini_batch_size}] "
                          f"p_loss={float(policy_loss.detach()):+.4f} "
                          f"v_loss={float(value_loss.detach()):+.4f} "
                          f"kl≈{ppo_metrics['approx_kl']:+.4f} "
                          f"clip_frac={ppo_metrics['clip_frac']:.2f}",
                          flush=True)

        n = max(1, agg["n_updates"])
        for k in ("policy_loss", "value_loss", "approx_kl", "clip_frac", "mean_ratio"):
            agg[k] /= n
        agg["n_samples"] = len(samples)
        return agg


# ---------------------------------------------------------------------------
# Sanity self-test (tiny mock)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Minimal test: verify the helper functions work on toy tensors."""
    print("[test] gather_log_probs")
    B, L, V = 2, 5, 10
    logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))
    lp = gather_log_probs(logits, labels)
    assert lp.shape == (B, L)
    assert (lp[:, 0] == 0).all()  # position 0 is placeholder
    # log_prob at position t should equal log_softmax(logits[t-1])[label[t]]
    for b in range(B):
        for t in range(1, L):
            expected = F.log_softmax(logits[b, t - 1], dim=-1)[labels[b, t]]
            assert torch.isclose(lp[b, t], expected, atol=1e-5), \
                f"mismatch at (b={b},t={t}): got {lp[b,t]} expected {expected}"
    print(f"  gather_log_probs: OK ({B}x{L})")

    print("[test] ppo_clip_loss shape + gradient")
    lp_new = torch.randn(B, L, requires_grad=True)
    lp_old = torch.randn(B, L)
    adv = torch.randn(B, L)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, 0] = False  # simulate prompt
    loss, m = ppo_clip_loss(lp_new, lp_old, adv, mask, ppo_clip=0.2)
    assert loss.dim() == 0
    loss.backward()
    assert lp_new.grad is not None and (lp_new.grad != 0).any()
    print(f"  loss={float(loss):+.4f}  kl≈{m['approx_kl']:+.4f}  "
          f"clip_frac={m['clip_frac']:.2f}  mean_ratio={m['mean_ratio']:.3f}")

    print("[test] ppo_clip_loss zero mask → zero loss + no grad crash")
    lp2 = torch.randn(B, L, requires_grad=True)
    zero_mask = torch.zeros(B, L, dtype=torch.bool)
    loss0, _ = ppo_clip_loss(lp2, lp_old, adv, zero_mask, ppo_clip=0.2)
    # loss should be 0 (no unmasked positions); backward should be safe
    assert float(loss0) == 0.0
    loss0.backward()
    print(f"  zero-mask loss OK: {float(loss0):+.4f}")

    print("[test] pad_batch")
    ts = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
    padded = pad_batch(ts, pad_value=0)
    assert padded.shape == (3, 3)
    assert (padded[0] == torch.tensor([1, 2, 3])).all()
    assert (padded[1] == torch.tensor([4, 5, 0])).all()
    assert (padded[2] == torch.tensor([6, 0, 0])).all()
    print(f"  pad_batch shape: {tuple(padded.shape)}  OK")

    print("[test] tokenize_prompt_and_response (with mock tokenizer)")
    class MockTok:
        pad_token_id = 0
        def __call__(self, text, add_special_tokens=False):
            # word-level tokenizer with padding-friendly output
            words = text.split() if text else []
            return {"input_ids": [i + 1 for i in range(len(words))]}  # avoid 0
        def decode(self, ids, skip_special_tokens=False):
            return " ".join(str(i) for i in ids)

    tk = MockTok()
    out = tokenize_prompt_and_response(tk, "hello world", "foo bar baz", max_length=100)
    print(f"  prompt_len={out['prompt_len']}  response_ids={out['response_ids']}  "
          f"input_ids={out['input_ids'].tolist()}")
    assert out["prompt_len"] == 2
    assert len(out["response_ids"]) == 3
    assert out["input_ids"].shape[0] == 5

    print("[test] all cf_content_trainer helper tests passed")


if __name__ == "__main__":
    _self_test()
