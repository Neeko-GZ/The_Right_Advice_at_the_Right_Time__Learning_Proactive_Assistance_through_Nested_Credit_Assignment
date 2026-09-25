"""
offline_eval.py — full-episode eval with the companion loaded LOCALLY
(AutoModelForCausalLM + PeftModel), bypassing the broken vLLM LoRA serving.

Advisee still runs on vLLM 8000 (that one works fine — no LoRA). We only
replace the companion with a local model so the trained LoRA actually applies.

Reports success rate + HELP rate over N episodes — the real numbers.

Usage (kill vLLM 8001 first to free GPU):
  PYTHONPATH=. python3 rl_causal/scripts/offline_eval.py \
      --base /workspace/models/Qwen3.5-9B \
      --adapter /workspace/checkpoints/cf_grpo_p2_v8_c10/ckpt_step_0020 \
      --advisee-url http://localhost:8000 --advisee-model /workspace/models/Qwen3.5-4B \
      --n-episodes 30 --max-steps 30 --seed 1234
"""

from __future__ import annotations

import argparse
import sys

import torch

from rl_causal.alfworld_env import ALFWorldEnv
from rl_causal.advisee import VllmAdvisee
from rl_causal.prompts.alfworld import SYSTEM_PROMPT_COMPANION, build_companion_prompt
from rl_causal.scripts.collect_rollouts import rollout_one_episode, _parse_gate_advice


class NullCompanion:
    """No companion at all — advisee plays alone (always SILENCE, no advice).
    The 'no-companion' baseline. Loads no model, so it is instant."""

    def act(self, obs, history, seed=None):
        return "SILENCE", "", True


class OfflineCompanion:
    def __init__(self, base, adapter, temperature=0.0, max_new_tokens=200, history_window=10,
                 force_gate=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
        )
        print(f"[companion] class={type(m).__name__}", flush=True)
        if adapter and adapter.lower() != "base":
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter)
            print(f"[companion] adapter applied: {adapter}", flush=True)
        else:
            print("[companion] BASE (no adapter)", flush=True)
        m.eval()
        self.model = m
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.hw = history_window
        # force_gate: None -> use model's gate; "help"/"silence" -> override gate
        # (advice still comes from the model, so we measure the content head
        #  independently of the learned gate).
        self.force_gate = force_gate.upper() if force_gate else None

    _call = 0

    @torch.no_grad()
    def act(self, obs, history, seed=None):
        import time as _t
        _t0 = _t.time()
        user = build_companion_prompt(obs, history=history, history_window=self.hw)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_COMPANION},
            {"role": "user", "content": user},
        ]
        # no-think: match TRAINING (rollout used enable_thinking=False) + much faster.
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.model.device)
        gkw = dict(max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.eos_token_id)
        if self.temperature > 0:
            gkw.update(do_sample=True, temperature=self.temperature, top_p=0.95)
        else:
            gkw.update(do_sample=False)
        out = self.model.generate(ids, **gkw)
        gen = self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        # Robust parse: gate literal appears first in the JSON, so it's intact
        # even if `advice` gets truncated by max_new_tokens (balanced-JSON
        # parsing would fail on truncation -> gate '?').
        import re as _re
        gm = _re.search(r'"gate"\s*:\s*"([A-Za-z]+)"', gen)
        am = _re.search(r'"advice"\s*:\s*"([^"]*)', gen)
        gate = gm.group(1).strip().upper() if gm else "SILENCE"
        advice = am.group(1) if am else ""
        ok = gm is not None
        if self.force_gate is not None:
            gate = self.force_gate  # override learned gate; keep model advice
        OfflineCompanion._call += 1
        print(f"      [act {OfflineCompanion._call}] gate={gate} "
              f"({_t.time()-_t0:.1f}s, {ids.shape[1]} in-tok)", flush=True)
        return gate, advice, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/models/Qwen3.5-9B")
    ap.add_argument("--adapter", required=True, help="'base' or a checkpoint path")
    ap.add_argument("--advisee-url", default="http://localhost:8000")
    ap.add_argument("--advisee-model", default="/workspace/models/Qwen3.5-4B")
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--force-gate", choices=["help", "silence"], default=None,
                    help="override the learned gate (advice still from model); "
                         "'help' = measure content head independent of gate")
    ap.add_argument("--advisee-backend", choices=["vllm", "openai"], default="vllm",
                    help="'openai' uses an API player (4o-mini / DeepSeek). Set "
                         "--advisee-model to the API model name, --advisee-api-base + "
                         "--advisee-api-key-env for the provider.")
    ap.add_argument("--advisee-api-base", default=None,
                    help="OpenAI-compatible base_url (e.g. https://api.deepseek.com)")
    ap.add_argument("--advisee-api-key-env", default="OPENAI_API_KEY")
    args = ap.parse_args()

    env = ALFWorldEnv()
    if args.advisee_backend == "openai":
        from rl_causal.advisee import OpenAIAdvisee
        advisee = OpenAIAdvisee(model=args.advisee_model, base_url=args.advisee_api_base,
                                api_key_env=args.advisee_api_key_env,
                                temperature=args.temperature)
        print(f"[advisee] API player: {args.advisee_model} "
              f"(base={args.advisee_api_base or 'openai'})", flush=True)
    else:
        advisee = VllmAdvisee(model=args.advisee_model, url=args.advisee_url)
    if args.adapter.lower() in ("silence", "none", "no-companion", "advisee-only"):
        companion = NullCompanion()
        print("[companion] NONE — advisee plays alone (no-companion baseline)", flush=True)
    else:
        companion = OfflineCompanion(args.base, args.adapter, temperature=args.temperature,
                                     force_gate=args.force_gate)
        if args.force_gate:
            print(f"[companion] FORCE GATE = {args.force_gate.upper()} "
                  f"(advice still from model)", flush=True)

    n_success = 0
    n_help = 0
    n_steps = 0
    for i in range(args.n_episodes):
        if hasattr(advisee, "reset_history"):
            advisee.reset_history()
        steps = rollout_one_episode(env, companion, advisee,
                                    max_steps=args.max_steps, seed=args.seed + i)
        won = any(s["r_env"] > 0.5 for s in steps)
        h = sum(1 for s in steps if s["gate"] == "HELP")
        n_success += int(won)
        n_help += h
        n_steps += len(steps)
        print(f"  ep {i+1}/{args.n_episodes}: {'WIN ' if won else 'fail'} "
              f"len={len(steps)} help={h} ({100*h/max(1,len(steps)):.0f}%)", flush=True)

    print("=" * 64)
    print(f"  adapter        : {args.adapter}")
    print(f"  seed / force   : {args.seed} / {args.force_gate or 'model-gate'}")
    print(f"  success rate   : {n_success}/{args.n_episodes} "
          f"({100*n_success/args.n_episodes:.1f}%)")
    print(f"  HELP rate      : {n_help}/{n_steps} ({100*n_help/max(1,n_steps):.1f}%)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
