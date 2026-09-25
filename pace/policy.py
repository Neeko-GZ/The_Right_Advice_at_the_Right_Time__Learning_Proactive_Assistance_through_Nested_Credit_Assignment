"""
policy.py - Qwen3.5-VL + SPARK + LoRA policy for PPO.

Module surface (matches rl/policy.py so ppo/rollout.py + losses.py reuse):

    policy = Policy(qwen_path, spark_ckpt, ...)
    decision, p_help = policy.decide(features)
    advice, gen_info = policy.generate_advice(features, task=task)
    logits, response_start = policy.forward_logits_with_response(
        features, response_ids, prompt_kind, task)

`features` is the rich observation bundle passed from MindcraftEnv:
    frames: list[PIL.Image]    # 16 RGB frames
    task: str
    state_compact: dict        # held / inv / nearby / vitals
    env_text: str | None       # current clip_text (may be None)

What's loaded
-------------
SPARK (frozen)          : 16-frame video → (task_emb, fused_emb) ∈ R^{512}
SparkProjector (train)  : 512 → 4096, two parallel projections
Qwen3.5-VL (base frozen): hidden_size 4096, vision tower for 1 RGB frame
LoRA (train)            : on q/k/v/o_proj of Qwen attention, rank 16
ValueHead (train)       : MLP critic on concat(task_emb, fused_emb) ∈ R^{1024}

Total trainable params (rough): LoRA ~12M + projector ~4M + value_head ~0.3M
≈ 16M trainable vs 9B+ total. The 99% frozen part doesn't accumulate grads.

Prompt structure
----------------
We compose three sources at the inputs_embeds level (no token-id round-trip
for the visual / soft tokens):

    [SYS prompt text] [IMG placeholder] [SPARK_TASK ph] [SPARK_FUSED ph]
    Task: <task>
    State: <state_compact serialized>
    Decision:

At forward time:
  * IMG placeholder embedding ← Qwen vision_tower(frames[-1])
  * SPARK_TASK ph embedding ← projector.task_proj(task_emb)
  * SPARK_FUSED ph embedding ← projector.fused_proj(fused_emb)

For HELP / SILENCE, we read the logits at the last position of the prompt
and gate the two specific token IDs. For advice text, we generate from the
prompt up to EOS with sampling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from rl_spark.spark_projector import SparkProjector, build_spark_projector
from rl_spark.spark_scorer import load_pretrained_spark, _head_forward_with_embs
from rl_spark.ppo.value_head import ValueHead, build_value_head

log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# config + constants
# ---------------------------------------------------------------------------

# Special string placeholders we insert in the prompt template; they get
# tokenized to ids that we then locate in input_ids to do the embedding
# replacement. The strings must be tokens that Qwen's tokenizer treats as
# single tokens — using <|reserved_xxx|> from Qwen's reserved vocab is the
# safe choice. We discover the actual reserved tokens at __init__ time.
DEFAULT_SPARK_TASK_PLACEHOLDER = "<|reserved_special_token_250|>"
DEFAULT_SPARK_FUSED_PLACEHOLDER = "<|reserved_special_token_251|>"


@dataclass
class PolicyConfig:
    qwen_path: str = "/workspace/models/Qwen3.5-9B"
    spark_trainable_ckpt: str = "checkpoints/spark_pretrain_moga/spark_best.pt"
    mineclip_ckpt: str = "/workspace/MineCLIP/attn.pth"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Value head
    value_input_dim: int = 1024  # 512+512 concat
    value_hidden: int = 256
    value_depth: int = 2

    # Prompt
    spark_task_placeholder: str = DEFAULT_SPARK_TASK_PLACEHOLDER
    spark_fused_placeholder: str = DEFAULT_SPARK_FUSED_PLACEHOLDER
    # Decision tokens: short, single-BPE-token, lowercase forms so the
    # tokenizer doesn't split them across multiple subwords. Earlier
    # "SILENCE"/"HELP" failed because "SILENCE" tokenized to 4 subwords
    # → silence_id pointed at "CE", which the model never predicts as
    # the first token. "yes"/"no" are essentially guaranteed to be
    # single tokens in any BPE vocabulary trained on English text.
    silence_text: str = "no"
    help_text: str = "yes"

    # Generation
    advice_max_new_tokens: int = 50
    # Minimum body length after the forced "yes" token. Without this,
    # the model can emit yes<EOS> immediately, leaving an empty advice
    # body and wasting the HELP step (bot gets empty message, PPO still
    # uses the sample with logp(yes) but no advice tokens for ratio).
    # Setting min=5 ensures the model commits to at least a short
    # sentence; rare cases where the model truly has nothing to say
    # would prefer SILENCE anyway (decided upstream by decide()).
    advice_min_new_tokens: int = 5
    advice_temperature: float = 0.7
    advice_top_p: float = 0.9

    # Decision-token bias (kept as a knob; default 0 since with the
    # "yes"/"no" decision tokens the base-model prior should be roughly
    # balanced, unlike the old "SILENCE"/"HELP" pair which saturated
    # heavily toward HELP. If you observe p_help still pinned near 0 or 1
    # in early rollouts, set this to ±a few nats. Applied consistently
    # in decide(), generate_advice() (via LogitsProcessor), and
    # forward_logits_with_response() so the PPO ratio remains correct
    # (the bias cancels in log_ratio between rollout and update).
    decide_silence_bias: float = 0.0

    # Runtime
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16


# Prompt template. State is rendered as compact key=value pairs in
# `_render_state` to keep the token count low.
PROMPT_TEMPLATE = """You are a Minecraft companion advising a survival agent.
{spark_task_placeholder}{spark_fused_placeholder}
Task: {task}
State: {state}
Last advice: {last_advice}
Based on the scene and the alignment signal above, decide whether the agent needs help right now.
Reply 'yes' followed by a one-sentence concrete suggestion if help is needed, or 'no' if not.
Reply:"""


def _render_state(state_compact: dict) -> str:
    """Compact key=value rendering for the state portion of the prompt."""
    if not state_compact:
        return "unknown"
    held = state_compact.get("held_item") or {}
    held_str = f"{held.get('name', 'none')}x{held.get('count', 0)}" if held else "none"

    inv = state_compact.get("inventory_top") or []
    inv_str = ",".join(f"{it['name']}x{it['count']}" for it in inv[:5]) or "empty"

    near = state_compact.get("nearby_top") or []
    near_str = ",".join(f"{e['name']}@{e['distance']:.1f}m" for e in near[:3]) or "none"

    hp = state_compact.get("health")
    food = state_compact.get("food")
    return f"held={held_str} inv=[{inv_str}] near=[{near_str}] hp={hp} food={food}"


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class Policy:
    """Qwen3.5-VL + SPARK + LoRA companion policy."""

    def __init__(self, cfg: Optional[PolicyConfig] = None, **kwargs):
        self.cfg = cfg or PolicyConfig(**kwargs)
        self.device = torch.device(self.cfg.device)

        log.info("loading Qwen3.5-VL...")
        self._build_qwen()

        log.info("loading SPARK (frozen)...")
        self.spark = load_pretrained_spark(
            mineclip_ckpt=self.cfg.mineclip_ckpt,
            trainable_ckpt=self.cfg.spark_trainable_ckpt,
            device=str(self.device),
            dtype=torch.float32,  # SPARK in fp32, projector handles cast to bf16
        )

        log.info("building projector + value head...")
        self.projector = build_spark_projector(
            spark_dim=512, qwen_hidden=4096,
            device=str(self.device), dtype=self.cfg.dtype,
        )
        self.value_head = build_value_head(
            input_dim=self.cfg.value_input_dim,
            hidden=self.cfg.value_hidden,
            depth=self.cfg.value_depth,
            device=str(self.device),
            dtype=torch.float32,
        )

        # Lookup token ids for placeholders / decision words
        self._resolve_token_ids()

        n_train = sum(
            p.numel() for p in self.trainable_parameters()
        )
        log.info(f"trainable params: {n_train:,}")

    # ----- model building helpers -----

    def _build_qwen(self):
        """Load Qwen3.5-VL base + apply LoRA. Cascade auto-class resolution
        like in spark_scorer for transformers version compatibility."""
        import transformers
        from transformers import AutoProcessor

        candidates = [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ]
        AutoVLM = None
        for name in candidates:
            cls = getattr(transformers, name, None)
            if cls is not None:
                AutoVLM = cls
                log.info(f"  using {name}")
                break
        if AutoVLM is None:
            raise RuntimeError("No VLM auto-class found in transformers")

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.qwen_path, trust_remote_code=True,
        )
        self.tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer") else self.processor
        )

        qwen = AutoVLM.from_pretrained(
            self.cfg.qwen_path,
            torch_dtype=self.cfg.dtype,
            device_map=str(self.device),
            trust_remote_code=True,
        )

        # Freeze base, then wrap with LoRA
        for p in qwen.parameters():
            p.requires_grad = False

        from peft import LoraConfig, get_peft_model, TaskType
        lora_cfg = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=list(self.cfg.lora_target_modules),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.qwen = get_peft_model(qwen, lora_cfg)

        # The wrapper enables grad only on LoRA params. We default to
        # eval mode + no gradient checkpointing so rollout-time forwards
        # (decide / generate_advice) run with KV cache and no dropout.
        # forward_logits_with_response toggles train mode + optionally
        # gradient checkpointing for PPO updates.
        self.qwen.eval()

        # Register SPARK soft-token injection hook on first decoder layer.
        # Placeholder rationale: when activated by callers (via the
        # _inject_state attribute), the hook runs AFTER Qwen has done its
        # internal embedding lookup + image masked_scatter_, so we never
        # interact with the in-place ops that broke earlier attempts.
        self._inject_state: Optional[dict] = None
        self._spark_hook_handle = self._register_spark_hook()

    def _resolve_token_ids(self):
        """Find token ids for the placeholders + decision words. Use
        `add_special_tokens=False` so we don't get BOS/EOS in the way."""
        tok = self.tokenizer

        def _single_token_or_none(s: str) -> Optional[int]:
            ids = tok(s, add_special_tokens=False).input_ids
            return ids[0] if len(ids) == 1 else None

        # Try the configured placeholder strings first. If they don't
        # tokenize to a single token (e.g. Qwen3.5 doesn't have
        # `<|reserved_special_token_*|>`), fall back to ADDING two new
        # special tokens to the tokenizer and resizing Qwen's embedding
        # matrix. The new embedding rows start random but will be
        # immediately overwritten by projector output at forward time, so
        # the initial values don't matter.
        task_id = _single_token_or_none(self.cfg.spark_task_placeholder)
        fused_id = _single_token_or_none(self.cfg.spark_fused_placeholder)

        if task_id is None or fused_id is None:
            log.warning(
                f"placeholder strings {self.cfg.spark_task_placeholder!r} / "
                f"{self.cfg.spark_fused_placeholder!r} don't tokenize cleanly; "
                f"adding fresh special tokens."
            )
            # NB: many models (Qwen3.5-VL included) pad their embedding
            # table beyond the tokenizer's nominal vocab size for GPU
            # efficiency (e.g. round up to a multiple of 64). Calling
            # `resize_token_embeddings(len(tok))` would TRUNCATE those
            # padded slots and corrupt the lm_head, leading to garbage
            # generation. We only resize when the newly assigned token ids
            # exceed the current embedding table.
            inner = getattr(self.qwen, "base_model", self.qwen)
            inner = getattr(inner, "model", inner)
            cur_vocab = inner.get_input_embeddings().num_embeddings

            new_specials = ["<|spark_task|>", "<|spark_fused|>"]
            added = tok.add_special_tokens(
                {"additional_special_tokens": new_specials}
            )
            log.info(f"  added {added} new special tokens to tokenizer")

            new_ids = [tok.convert_tokens_to_ids(s) for s in new_specials]
            max_new_id = max(new_ids)
            if max_new_id >= cur_vocab:
                # Genuinely need more rows; expand without shrinking.
                inner.resize_token_embeddings(max_new_id + 1)
                log.info(
                    f"  resized embeddings: {cur_vocab} -> {max_new_id + 1}"
                )
            else:
                log.info(
                    f"  new token ids {new_ids} fit in existing embedding "
                    f"({cur_vocab} rows) — no resize needed"
                )

            # Update config placeholders to the new strings + look up ids
            self.cfg.spark_task_placeholder = new_specials[0]
            self.cfg.spark_fused_placeholder = new_specials[1]
            task_id = _single_token_or_none(new_specials[0])
            fused_id = _single_token_or_none(new_specials[1])
            if task_id is None or fused_id is None:
                raise RuntimeError(
                    "Failed to register spark placeholder tokens even after add_special_tokens"
                )

        self.spark_task_id = task_id
        self.spark_fused_id = fused_id
        log.info(f"  spark_task placeholder id = {self.spark_task_id}")
        log.info(f"  spark_fused placeholder id = {self.spark_fused_id}")

        # Decision tokens: must be single BPE tokens so the lm_head logit
        # at silence_id / help_id is what the model would actually predict
        # as the FIRST token after the prompt's "Reply:". If multi-token,
        # silence_id would point at a subword the model never generates
        # first → decide always saturated. Assert catches this loudly.
        silence_ids = tok(self.cfg.silence_text, add_special_tokens=False).input_ids
        help_ids = tok(self.cfg.help_text, add_special_tokens=False).input_ids
        assert len(silence_ids) == 1, (
            f"silence_text {self.cfg.silence_text!r} tokenizes to {len(silence_ids)} pieces "
            f"({silence_ids}); decide head needs single-token decision words"
        )
        assert len(help_ids) == 1, (
            f"help_text {self.cfg.help_text!r} tokenizes to {len(help_ids)} pieces "
            f"({help_ids}); decide head needs single-token decision words"
        )
        self.silence_id = silence_ids[-1]
        self.help_id = help_ids[-1]
        log.info(f"  silence_id = {self.silence_id}, help_id = {self.help_id}")
        # DIAGNOSTIC: dump the full subword tokenization so we can see
        # if 'SILENCE' / 'HELP' tokenize to multiple BPE pieces. If multi,
        # silence_id/help_id are LAST subwords and our decide/bias logic
        # is biasing the wrong token.
        log.info(
            f"  silence tokenization: ids={silence_ids} "
            f"strs={[tok.decode([i]) for i in silence_ids]}"
        )
        log.info(
            f"  help tokenization: ids={help_ids} "
            f"strs={[tok.decode([i]) for i in help_ids]}"
        )
        assert self.silence_id != self.help_id, "decision tokens must differ"

        # Resolve chat-turn-end token. Qwen3 emits <|im_end|> at the end of
        # every assistant turn; without including it in generate's
        # eos_token_id, the model occasionally continues past it and
        # hallucinates a new "User\n...assistant\n" turn — observed as
        # advice-string leak where the entire prompt template appears in
        # the generated advice.
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        unk_id = tok.unk_token_id
        if im_end_id is None or im_end_id == unk_id:
            log.warning(
                "  <|im_end|> not found in tokenizer; relying on "
                "default eos_token_id only — advice may sometimes leak "
                "prompt-template artifacts"
            )
            self.im_end_id = None
        else:
            self.im_end_id = int(im_end_id)
            log.info(f"  im_end_id = {self.im_end_id}")

    # ----- SPARK injection hook (forward_pre_hook on layer 0) -----

    def _find_first_decoder_layer(self):
        """Locate the first LANGUAGE-MODEL transformer decoder layer.

        Qwen3-VL is composed of (a) a vision tower and (b) a language
        model decoder; we want (b)'s first layer. The vision tower also
        has a `.layers` ModuleList (with self_attn + position_embeddings
        and cu_seqlens for varlen attention), so a naive search by
        `.layers` attribute can return the wrong stack. Heuristics:

          1. Prefer paths that explicitly contain 'language_model',
             'text_model', or 'lm' — these always refer to the LM
             decoder in HF model conventions.
          2. As a fallback, walk the module tree and pick the longest
             ModuleList whose path does NOT contain 'visual' / 'vision'.
          3. Final fallback: dump structure and raise.

        Returns the nn.Module to register the pre-hook on.
        """
        # Phase 1: explicit LM paths. Qwen3.5-VL uses an extra `.model.`
        # nesting between the PEFT wrap and language_model, so the actual
        # path is `base_model.model.model.language_model.layers`.
        lm_candidates = [
            "qwen.base_model.model.model.language_model.layers",
            "qwen.base_model.model.model.language_model.model.layers",
            "qwen.base_model.model.language_model.model.layers",
            "qwen.base_model.model.language_model.layers",
            "qwen.base_model.model.text_model.model.layers",
            "qwen.base_model.model.text_model.layers",
            "qwen.base_model.model.lm.layers",
            "qwen.model.model.language_model.layers",
            "qwen.model.language_model.model.layers",
            "qwen.model.language_model.layers",
            "qwen.model.text_model.layers",
        ]
        for path in lm_candidates:
            obj = self
            ok = True
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    ok = False
                    break
            if ok and hasattr(obj, "__len__") and len(obj) > 0:
                log.info(f"  SPARK hook target (LM): policy.{path}[0]")
                return obj[0]

        # Phase 2: walk module tree, exclude vision paths
        candidates_found = []
        for name, module in self.qwen.named_modules():
            lname = name.lower()
            if "visual" in lname or "vision" in lname or "patch_embed" in lname:
                continue
            try:
                if hasattr(module, "__len__") and len(module) > 0:
                    first = module[0]
                    # Qwen3.5 uses `linear_attn` for some layers; older
                    # models use `self_attn` / `attn` / `attention`.
                    if (
                        hasattr(first, "self_attn")
                        or hasattr(first, "attn")
                        or hasattr(first, "attention")
                        or hasattr(first, "linear_attn")
                    ):
                        candidates_found.append((name, len(module), type(first).__name__))
            except (TypeError, KeyError):
                continue
        if candidates_found:
            # Prefer the longest non-vision ModuleList (= the LM stack)
            candidates_found.sort(key=lambda x: -x[1])
            name, n, type_name = candidates_found[0]
            log.info(
                f"  SPARK hook target (LM-fallback): policy.qwen.{name}[0] "
                f"({n} layers of {type_name})"
            )
            obj = self.qwen
            for attr in name.split("."):
                obj = obj[int(attr)] if attr.isdigit() else getattr(obj, attr)
            return obj[0]

        # Last-ditch: dump structure
        log.error("Could not locate LM decoder layer. Module names containing 'layer':")
        for name, _ in self.qwen.named_modules():
            if "layer" in name.lower():
                log.error(f"    {name}")
        raise RuntimeError(
            "Could not locate first LM decoder layer in Qwen-VL model. "
            "See log above; add the correct path to _find_first_decoder_layer."
        )

    def _register_spark_hook(self):
        """Register a forward_pre_hook on the first decoder layer that
        splices SPARK soft tokens into hidden_states at the placeholder
        positions. The hook is a no-op unless self._inject_state is set.
        """
        layer0 = self._find_first_decoder_layer()

        # DIAGNOSTIC: count hook fires + whether they actually spliced.
        # Read via policy.spark_hook_stats from outside. Reset between runs.
        self.spark_hook_stats = {
            "fires": 0,
            "spliced_task": 0,
            "spliced_fused": 0,
            "skipped_state_none": 0,
            "skipped_T_too_small": 0,
            "skipped_no_positions": 0,
        }

        def hook(module, args, kwargs):
            stats = self.spark_hook_stats
            stats["fires"] += 1
            # On the very first fire, dump arg/kwarg structure so we can
            # confirm which slot holds (B, T, H) hidden_states.
            if stats["fires"] == 1:
                arg_shapes = []
                for i, a in enumerate(args):
                    if hasattr(a, "shape"):
                        arg_shapes.append(f"args[{i}]:{tuple(a.shape)}/{a.dtype}")
                    else:
                        arg_shapes.append(f"args[{i}]:{type(a).__name__}")
                kwarg_shapes = []
                for k, v in kwargs.items():
                    if hasattr(v, "shape"):
                        kwarg_shapes.append(f"{k}:{tuple(v.shape)}")
                    elif v is None:
                        kwarg_shapes.append(f"{k}:None")
                    else:
                        kwarg_shapes.append(f"{k}:{type(v).__name__}")
                log.info(
                    f"  [SPARK hook first-fire] args={arg_shapes} "
                    f"kwargs={kwarg_shapes}"
                )
            state = self._inject_state
            if state is None:
                stats["skipped_state_none"] += 1
                return None
            # Locate hidden_states: try args[0], then kwargs['hidden_states'],
            # then any 3D tensor in args (some Qwen versions pass it in a
            # different positional slot).
            hs = None
            from_args = False
            arg_idx = None
            if len(args) > 0 and hasattr(args[0], "dim") and args[0].dim() == 3:
                hs = args[0]
                from_args = True
                arg_idx = 0
            elif "hidden_states" in kwargs and kwargs["hidden_states"] is not None:
                hs = kwargs["hidden_states"]
                from_args = False
            else:
                for i, a in enumerate(args):
                    if hasattr(a, "dim") and a.dim() == 3:
                        hs = a
                        from_args = True
                        arg_idx = i
                        break
            if hs is None or hs.dim() != 3:
                stats.setdefault("skipped_no_hidden_states", 0)
                stats["skipped_no_hidden_states"] += 1
                return None
            B, T, H = hs.shape
            prompt_len = state["prompt_length"]
            # Only splice on the full-prompt forward, not on KV-cached
            # generation steps (where T == 1).
            if T < prompt_len:
                stats["skipped_T_too_small"] += 1
                return None
            task_pos = state["task_position"]
            fused_pos = state["fused_position"]
            if task_pos is None and fused_pos is None:
                stats["skipped_no_positions"] += 1
                return None
            soft = state["soft_tokens"].to(hs.dtype)  # (1, 2, H)
            position_ids = torch.arange(T, device=hs.device)
            new_hs = hs
            if task_pos is not None and 0 <= task_pos < T:
                mask_t = (position_ids == task_pos).view(1, T, 1)
                soft_t = soft[:, 0:1, :].expand(B, T, H)
                new_hs = torch.where(mask_t, soft_t, new_hs)
                stats["spliced_task"] += 1
            if fused_pos is not None and 0 <= fused_pos < T:
                mask_f = (position_ids == fused_pos).view(1, T, 1)
                soft_f = soft[:, 1:2, :].expand(B, T, H)
                new_hs = torch.where(mask_f, soft_f, new_hs)
                stats["spliced_fused"] += 1
            if from_args:
                new_args = list(args)
                new_args[arg_idx] = new_hs
                return tuple(new_args), kwargs
            else:
                return args, {**kwargs, "hidden_states": new_hs}

        return layer0.register_forward_pre_hook(hook, with_kwargs=True)

    def _inject_context(self, soft_tokens, task_position, fused_position, prompt_length):
        """Context manager that sets self._inject_state for the duration
        of the with-block. Caller passes the soft_tokens tensor (graph
        connected to projector) so projector receives gradients via the
        torch.where splice inside the hook.
        """
        policy = self

        class _Ctx:
            def __enter__(self_inner):
                policy._inject_state = {
                    "soft_tokens": soft_tokens,
                    "task_position": task_position,
                    "fused_position": fused_position,
                    "prompt_length": prompt_length,
                }

            def __exit__(self_inner, exc_type, exc_val, exc_tb):
                policy._inject_state = None
                return False

        return _Ctx()

    # ----- parameter inventory for the optimizer -----

    def trainable_parameters(self):
        """Iterator over params that require_grad=True."""
        for p in self.qwen.parameters():
            if p.requires_grad:
                yield p
        for p in self.projector.parameters():
            if p.requires_grad:
                yield p
        for p in self.value_head.parameters():
            if p.requires_grad:
                yield p

    # ----- inputs_embeds construction (heart of the file) -----

    def _spark_embeds(self, frames, task: str, env_text: Optional[str]):
        """Run SPARK once to get (task_emb, fused_emb) ∈ R^{512}. No grad."""
        env_t = env_text if env_text else "play minecraft"
        with torch.no_grad():
            # Preprocess frames to (1, T, 3, 160, 256) float on device
            clip = self._frames_to_clip_tensor(frames)
            img_feats = self.spark.forward_image_features(clip)
            video_feat = self.spark.forward_video_features(img_feats)
            _, task_emb, fused_emb = _head_forward_with_embs(
                self.spark.reward_head, video_feat, [task], [env_t],
            )
        return task_emb.float(), fused_emb.float()  # (1, 512), (1, 512)

    def _frames_to_clip_tensor(self, frames) -> torch.Tensor:
        """list[PIL.Image of any size] → (1, 16, 3, 160, 256) float on device."""
        from rl_spark.spark_scorer import WINDOW, TARGET_W, TARGET_H
        arrs = []
        if len(frames) < WINDOW:
            frames = [frames[0]] * (WINDOW - len(frames)) + list(frames)
        elif len(frames) > WINDOW:
            frames = frames[-WINDOW:]
        for img in frames:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if img.size != (TARGET_W, TARGET_H):
                img = img.resize((TARGET_W, TARGET_H), Image.BILINEAR)
            arrs.append(np.asarray(img, dtype=np.uint8))
        arr = np.stack(arrs, axis=0)  # (T, H, W, 3)
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # (T, 3, H, W)
        return t.unsqueeze(0).to(self.device)  # (1, T, 3, H, W)

    def _build_prompt(self, task: str, state_compact: dict,
                      last_advice: Optional[str]) -> str:
        """Prompt body WITHOUT image placeholder. The processor's chat
        template inserts the correct number of <|image_pad|> tokens to
        match the visual encoder's output, so we don't manage that here."""
        return PROMPT_TEMPLATE.format(
            spark_task_placeholder=self.cfg.spark_task_placeholder,
            spark_fused_placeholder=self.cfg.spark_fused_placeholder,
            task=task,
            state=_render_state(state_compact),
            last_advice=(last_advice or "(none)"),
        )

    def _build_proc_inputs(self, features: dict) -> dict:
        """Build the multimodal input batch via processor.

        Standard path: pass input_ids + pixel_values + image_grid_thw +
        mm_token_type_ids to model.forward / generate, let Qwen3-VL handle
        image embedding + splicing internally.

        SPARK soft-token injection now ACTIVE via a forward_pre_hook on
        the first decoder layer (see _register_spark_hook). The hook runs
        AFTER Qwen's internal masked_scatter_ for image splice, so we
        avoid the in-place autograd version conflict that broke earlier
        attempts (embedding-layer hook + manual inputs_embeds).

        Caller wraps the model forward in self._inject_context(...) using
        the soft_tokens / task_position / fused_position / prompt_length
        keys returned here.

        Returns:
            input_ids         : (1, T)
            attention_mask    : (1, T)
            pixel_values      : (N_patch, D_pix)
            image_grid_thw    : (1, 3)
            mm_token_type_ids : (1, T) or None
            prompt_length     : int
            soft_tokens       : (1, 2, H)  — projector output, has grad
            task_position     : int or None  — index of spark_task_id in input_ids
            fused_position    : int or None  — index of spark_fused_id in input_ids
        """
        frames = features["frames"]
        task = features["task"]
        state = features.get("state_compact") or {}
        env_text = features.get("env_text")
        last_advice = features.get("last_advice")

        # SPARK soft tokens — computed for future re-injection. Currently
        # NOT spliced into the model input (see docstring).
        task_emb, fused_emb = self._spark_embeds(frames, task, env_text)
        soft_tokens = self.projector(
            task_emb.to(self.cfg.dtype),
            fused_emb.to(self.cfg.dtype),
        )  # (1, 2, H)

        # Build chat-templated text; processor expands {"type": "image"}
        # into the right number of <|image_pad|> tokens.
        prompt_body = self._build_prompt(task, state, last_advice)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_body},
            ],
        }]
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        inputs = self.processor(
            text=text, images=[frames[-1]], return_tensors="pt",
        )
        mm_token_type_ids = getattr(inputs, "mm_token_type_ids", None)
        if mm_token_type_ids is None and isinstance(inputs, dict):
            mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(self.device)

        # Locate spark placeholder positions in the prompt's input_ids
        # (the hook uses these to splice soft_tokens at the correct
        # indices in hidden_states before the first decoder layer).
        ids_list = inputs.input_ids[0].tolist()
        task_position = (
            ids_list.index(self.spark_task_id)
            if self.spark_task_id in ids_list else None
        )
        fused_position = (
            ids_list.index(self.spark_fused_id)
            if self.spark_fused_id in ids_list else None
        )

        return {
            "input_ids":         inputs.input_ids.to(self.device),
            "attention_mask":    inputs.attention_mask.to(self.device),
            "pixel_values":      inputs.pixel_values.to(self.device, dtype=self.cfg.dtype),
            "image_grid_thw":    inputs.image_grid_thw.to(self.device),
            "mm_token_type_ids": mm_token_type_ids,
            "prompt_length":     inputs.input_ids.shape[1],
            "soft_tokens":       soft_tokens,
            "task_position":     task_position,
            "fused_position":    fused_position,
        }

    # ----- decision head -----

    @torch.no_grad()
    def decide(self, features: dict) -> tuple[str, float]:
        """Return (decision, p_help). decision is 'HELP' or 'SILENCE'."""
        self.qwen.eval()
        enc = self._build_proc_inputs(features)
        with self._inject_context(
            soft_tokens=enc["soft_tokens"],
            task_position=enc["task_position"],
            fused_position=enc["fused_position"],
            prompt_length=enc["prompt_length"],
        ):
            out = self.qwen(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                pixel_values=enc["pixel_values"],
                image_grid_thw=enc["image_grid_thw"],
                mm_token_type_ids=enc["mm_token_type_ids"],
                use_cache=False,
            )
        last_logits = out.logits[:, -1, :].float()  # (1, V)
        # Apply silence bias to rebalance saturated prior (see PolicyConfig
        # docstring). Functional, not in-place, to avoid autograd issues.
        bias = float(self.cfg.decide_silence_bias)
        if bias != 0.0:
            last_logits = last_logits.clone()
            last_logits[:, self.silence_id] = last_logits[:, self.silence_id] + bias
        binary_logits = last_logits[:, [self.silence_id, self.help_id]]  # (1, 2)
        p_help = float(F.softmax(binary_logits, dim=-1)[0, 1].item())
        decision = "HELP" if p_help >= 0.5 else "SILENCE"
        return decision, p_help

    # ----- advice generation -----

    @torch.no_grad()
    def generate_advice(self, features: dict, task: str) -> tuple[str, dict]:
        """Sample advice text given the features.

        Returns (advice_text, info). info contains:
            new_ids: list[int]  — response tokens (after prompt)
            prompt_length: int
        """
        self.qwen.eval()
        enc = self._build_proc_inputs(features)
        prompt_length = enc["prompt_length"]

        # Force the first response token to be "yes" (help_id) so the
        # decide/generate inconsistency is eliminated:
        #   - decide() chose HELP based on p_help > 0.5.
        #   - But free sampling often picked "no" or just "yes"<EOS>
        #     in the full-vocab distribution, wasting samples (bot got
        #     empty advice, PPO trained on weird HELP+'no' transitions).
        #   - By appending yes_id to input_ids before generate, we commit
        #     to the HELP path and let generate sample only the advice
        #     body (yes is in input, not sampled).
        # PPO consistency: we still include yes as response_ids[0] so the
        # logp at the decide position is part of the training signal —
        # the decide head and the first response token are now the same
        # quantity, learned jointly.
        yes_id = self.help_id
        yes_tensor = torch.tensor(
            [[yes_id]], device=self.device, dtype=enc["input_ids"].dtype,
        )
        forced_input_ids = torch.cat([enc["input_ids"], yes_tensor], dim=1)
        forced_attn = torch.cat(
            [enc["attention_mask"], torch.ones_like(yes_tensor)], dim=1,
        )
        if enc["mm_token_type_ids"] is not None:
            forced_mm = torch.cat([
                enc["mm_token_type_ids"],
                torch.zeros_like(yes_tensor, dtype=enc["mm_token_type_ids"].dtype),
            ], dim=1)
        else:
            forced_mm = None
        forced_prompt_length = prompt_length + 1

        with self._inject_context(
            soft_tokens=enc["soft_tokens"],
            task_position=enc["task_position"],
            fused_position=enc["fused_position"],
            prompt_length=enc["prompt_length"],  # SPARK placeholders are
                                                  # in the ORIGINAL prompt
                                                  # (before the forced yes)
        ):
            # Stop on either model EOS or chat-turn-end. Without
            # <|im_end|> in eos_token_id, Qwen3 occasionally generates
            # past the assistant turn and hallucinates a new
            # "User\n...assistant\n<think>...</think>\n\nyes" sequence
            # that leaks into advice.
            eos_token_ids: list[int] = [int(self.tokenizer.eos_token_id)]
            if self.im_end_id is not None and self.im_end_id not in eos_token_ids:
                eos_token_ids.append(self.im_end_id)

            gen_out = self.qwen.generate(
                input_ids=forced_input_ids,
                attention_mask=forced_attn,
                pixel_values=enc["pixel_values"],
                image_grid_thw=enc["image_grid_thw"],
                mm_token_type_ids=forced_mm,
                max_new_tokens=self.cfg.advice_max_new_tokens,
                # min_new_tokens prevents the model from terminating
                # immediately after the forced "yes" token (which would
                # leave an empty advice body, wasting the HELP step).
                min_new_tokens=self.cfg.advice_min_new_tokens,
                do_sample=True,
                temperature=self.cfg.advice_temperature,
                top_p=self.cfg.advice_top_p,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=eos_token_ids,
                # Note: no longer using a decide_silence_bias LogitsProcessor.
                # The bias only mattered for sampling yes vs no at the
                # decision position — now we've forced yes, so any bias on
                # the silence_id at this position is irrelevant for generate.
                # (The bias still applies in decide() and in
                # forward_logits_with_response()'s logits at the decision
                # position, where it remains consistent for PPO ratio.)
            )

        # gen_out.sequences = [prompt + forced_yes + body]. The advice body
        # is what came after the forced yes.
        seq = gen_out.sequences
        if seq.shape[1] > forced_prompt_length:
            body_ids = seq[0, forced_prompt_length:].tolist()
        else:
            body_ids = []
        # advice text = body only (no leading "yes"). Strip leading
        # punctuation that's the natural continuation after "yes" in
        # English ("yes," "yes." "yes—" "yes:"), so the bot sees a
        # clean sentence rather than ", use the axe ..." etc.
        advice = self.tokenizer.decode(body_ids, skip_special_tokens=True).strip()
        advice = advice.lstrip(",.!?;:—– \t\n")
        # Defense-in-depth: if eos_token_id failed to stop generation
        # and the model still emitted a new chat turn, truncate at the
        # earliest such marker. These are the strings Qwen3's chat
        # template uses verbatim when the model hallucinates a new turn.
        for stop_marker in ("\nUser\n", "\nuser\n", "\nassistant\n",
                            "<|im_start|>", "<|im_end|>", "<think>", "</think>"):
            idx = advice.find(stop_marker)
            if idx >= 0:
                advice = advice[:idx].rstrip()
        # Capitalize first letter for readability if it was lowercased
        # (e.g. "use your axe" → "Use your axe").
        if advice and advice[0].islower():
            advice = advice[0].upper() + advice[1:]
        # response_ids for PPO: include the forced yes so the model is
        # trained on logp(yes | prompt) at the decision position.
        # prompt_length stays at the ORIGINAL value: the first response
        # token is yes, and gather_response_logprobs picks logits at
        # positions [prompt_length-1 ...] which predicts response_ids[0] = yes.
        response_ids = [yes_id] + body_ids
        return advice, {"new_ids": response_ids, "prompt_length": prompt_length}

    # ----- forward for PPO update (with grad) -----

    def forward_logits_with_response(
            self,
            features: dict,
            response_ids_t: torch.Tensor,  # (1, R) long on device
            prompt_kind: str = "advice",
            task: Optional[str] = None,
    ) -> tuple[torch.Tensor, int]:
        """Re-forward with the same prompt + a fixed response sequence.

        Returns (logits, response_start) where:
            logits: (1, T+R, V) — full sequence logits
            response_start: int — index in T+R where the response begins;
                logits[:, response_start - 1 + i] predicts response_ids[:, i].
        """
        # PPO update needs training mode (for LoRA + projector grad flow);
        # caller is responsible for switching back to eval() if doing more
        # rollouts afterwards. We don't toggle gradient checkpointing here
        # — enable it externally if memory pressure demands it.
        self.qwen.train()
        enc = self._build_proc_inputs(features)
        prompt_length = enc["prompt_length"]

        # Extend input_ids + attention_mask + mm_token_type_ids with the
        # response. Response tokens are pure text → mm_token_type_ids = 0
        # for those positions.
        if response_ids_t.shape[1] == 0:
            full_input_ids = enc["input_ids"]
            full_attention_mask = enc["attention_mask"]
            full_mm_token_type_ids = enc["mm_token_type_ids"]
        else:
            full_input_ids = torch.cat([enc["input_ids"], response_ids_t], dim=1)
            full_attention_mask = torch.cat(
                [enc["attention_mask"], torch.ones_like(response_ids_t)], dim=1,
            )
            if enc["mm_token_type_ids"] is None:
                full_mm_token_type_ids = None
            else:
                full_mm_token_type_ids = torch.cat(
                    [
                        enc["mm_token_type_ids"],
                        torch.zeros_like(response_ids_t,
                                         dtype=enc["mm_token_type_ids"].dtype),
                    ],
                    dim=1,
                )

        with self._inject_context(
            soft_tokens=enc["soft_tokens"],
            task_position=enc["task_position"],
            fused_position=enc["fused_position"],
            prompt_length=enc["prompt_length"],
        ):
            out = self.qwen(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                pixel_values=enc["pixel_values"],
                image_grid_thw=enc["image_grid_thw"],
                mm_token_type_ids=full_mm_token_type_ids,
                use_cache=False,
            )
        logits = out.logits
        # Apply silence bias at the position that predicts the decision
        # token (i.e., index prompt_length-1, which generates token at
        # index prompt_length = first response token). Done functionally
        # (additive) so autograd is preserved and bias cancels in the
        # PPO log-ratio.
        bias = float(self.cfg.decide_silence_bias)
        if bias != 0.0 and prompt_length >= 1:
            bias_tensor = torch.zeros_like(logits)
            bias_tensor[:, prompt_length - 1, self.silence_id] = bias
            logits = logits + bias_tensor
        return logits, prompt_length

    # ----- value head (PPO critic) -----

    def value(self, features: dict) -> torch.Tensor:
        """V(s) for PPO critic. Uses concat(task_emb, fused_emb)."""
        task_emb, fused_emb = self._spark_embeds(
            features["frames"], features["task"], features.get("env_text"),
        )
        x = torch.cat([task_emb, fused_emb], dim=-1).float()  # (1, 1024)
        return self.value_head(x).squeeze(0)  # scalar

    # ----- save / load -----

    def save_trainable(self, path):
        """Save LoRA + projector + value_head state dicts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        from peft import get_peft_model_state_dict
        torch.save(
            {
                "lora_state_dict": get_peft_model_state_dict(self.qwen),
                "projector_state_dict": self.projector.state_dict(),
                "value_head_state_dict": self.value_head.state_dict(),
                "config": self.cfg.__dict__,
            },
            path,
        )
        log.info(f"saved policy trainable state to {path}")

    def load_trainable(self, path):
        from peft import set_peft_model_state_dict
        payload = torch.load(path, map_location="cpu")
        set_peft_model_state_dict(self.qwen, payload["lora_state_dict"])
        self.projector.load_state_dict(payload["projector_state_dict"])
        self.value_head.load_state_dict(payload["value_head_state_dict"])
        log.info(f"loaded policy trainable state from {path}")
