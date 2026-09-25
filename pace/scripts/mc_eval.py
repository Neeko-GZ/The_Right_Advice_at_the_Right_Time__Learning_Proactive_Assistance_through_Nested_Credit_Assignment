"""
mc_eval.py — offline eval for the Minecraft NIC companion (VL, Option A).

Runs N episodes across the task curriculum with a FROZEN advisee and measures:
  * success rate (per task + overall)   — terminal milestone reached
  * HELP rate                            — fraction of steps the gate said HELP
  * milestone completion                 — mean fraction of task milestones reached

Three companion modes (the C4 comparison rows):
  --mode model-gate   trained companion decides gate + advice   (ours)
  --mode silence      no companion (advisee alone)              (no-companion floor)
  --mode force-help   gate always HELP (advice from companion)  (content-only / over-help)

Companion is served by vLLM :8001. Pass --adapter to hot-reload the eval
checkpoint into the 'companion' slot first (fresh VL LoRA needs no remap).

Run:
    cd /workspace/credit && PYTHONPATH=. python3 rl_causal/scripts/mc_eval.py \
      --spark-url http://172.17.0.1:18765 \
      --advisee-url http://localhost:8000 --advisee-model /workspace/models/Qwen3.5-4B \
      --companion-url http://localhost:8001 --companion-model companion \
      --adapter /workspace/checkpoints/mc_vl_v1/mc_ckpt_step_0010 \
      --mode model-gate --n-episodes 20
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict

from rl_causal.mc_env import MCEnv
from rl_causal.advisee import VllmAdvisee
from rl_causal.ppo.cf_rollout_worker import VllmCompanionRaw
from rl_causal.scripts.train_cf_grpo import hot_reload_companion_lora, remap_lora_for_vllm


CURRICULUM = ["collect_wood", "craft_crafting_table", "craft_wooden_pickaxe",
              "collect_stone", "craft_stone_pickaxe"]


import re as _re


def _parse_gate_advice_mc(raw):
    raw = raw or ""
    gm = _re.search(r'"gate"\s*:\s*"([A-Za-z]+)"', raw)
    am = _re.search(r'"advice"\s*:\s*"([^"]*)', raw)
    gate = gm.group(1).strip().upper() if gm else "SILENCE"
    advice = am.group(1).strip() if am else ""
    return gate, advice


class OpenAICompanionMC:
    """MC companion backed by an OpenAI-compatible API (e.g. gpt-4o-mini, VL).
    Sends the player-view frame(s) as image_url via prompts.mc.build_companion_prompt.
    Same act() signature as VllmCompanionRaw."""

    def __init__(self, model="gpt-4o-mini", base_url=None, api_key_env="OPENAI_API_KEY",
                 max_tokens=120, temperature=0.4, history_window=10):
        import os
        from openai import OpenAI
        from rl_causal.prompts.mc import SYSTEM_PROMPT_COMPANION, build_companion_prompt
        key = os.environ.get(api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"{api_key_env} env var not set")
        kw = {"api_key": key}
        if base_url:
            kw["base_url"] = base_url
        self._client = OpenAI(**kw)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history_window = history_window
        self._system = SYSTEM_PROMPT_COMPANION
        self._build = build_companion_prompt

    def act(self, obs, history, seed=None, temperature=None, top_p=None, focus=None):
        user = self._build(obs, history=history, history_window=self.history_window, focus=focus)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self._system},
                      {"role": "user", "content": user}],
            max_tokens=self.max_tokens,
            temperature=self.temperature if temperature is None else float(temperature),
        )
        raw = resp.choices[0].message.content or ""
        gate, advice = _parse_gate_advice_mc(raw)
        return gate, advice, raw


def run_episode(env, advisee, companion, mode, task, seed, verbose=False, help_interval=0):
    env.task_name = task
    obs, info = env.reset(seed=seed)
    advisee.reset_history()
    hist = []
    help_count = 0
    steps = 0
    won = False
    import random as _random
    _rng = _random.Random(seed)
    _next_help_at = 0   # random-step: index of the next forced HELP
    for t in range(env.max_steps + 2):
        g, adv = "SILENCE", ""
        if mode == "silence":
            gate, advice = "SILENCE", None
        elif mode == "fixed-step":
            # Fixed HELP schedule (ignores the learned gate): HELP every
            # (help_interval+1) steps, advice from the companion; else SILENCE.
            # help_interval=0 -> every step; =1 -> every 2nd; =3 -> every 4th; etc.
            if t % (help_interval + 1) == 0:
                g, adv, _ = companion.act(obs, hist, seed=seed * 100 + t, temperature=0.0)
                gate, advice = "HELP", adv
            else:
                gate, advice = "SILENCE", None
        elif mode == "random-step":
            # Random HELP schedule (ignores the learned gate): after each HELP,
            # wait a random gap of 0..help_interval SILENCE steps, then HELP again.
            # Advice from the companion; gate timing is random (seeded).
            if t >= _next_help_at:
                g, adv, _ = companion.act(obs, hist, seed=seed * 100 + t, temperature=0.0)
                gate, advice = "HELP", adv
                _next_help_at = t + 1 + _rng.randint(0, help_interval)
            else:
                gate, advice = "SILENCE", None
        else:
            g, adv, _ = companion.act(obs, hist, seed=seed * 100 + t, temperature=0.0)
            if mode == "force-help":
                gate, advice = "HELP", adv
            else:  # model-gate
                gate, advice = g, (adv if g == "HELP" else None)
        if gate == "HELP":
            help_count += 1
        action = advisee.act(obs, advice, seed=seed * 1000 + t)
        if verbose:
            inv = (obs.get("text") or "").replace("\n", " | ")
            print(f"    [t{t}] companion: gate={g} advice={adv!r}\n"
                  f"          player -> {action}\n"
                  f"          state: {inv[:200]}", flush=True)
        obs, r, done, trunc, info = env.step(action)
        hist.append((f"gate={gate}", action))
        steps = t + 1
        if info.get("won"):
            won = True
        if done or trunc:
            break
    n_ms = len(env._seen_milestones)
    total_ms = max(1, len(env.current_milestones))
    return {"task": task, "won": won, "help": help_count, "steps": steps,
            "ms_frac": n_ms / total_ms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark-url", default="http://172.17.0.1:18765")
    ap.add_argument("--advisee-url", default="http://localhost:8000")
    ap.add_argument("--advisee-model", default="/workspace/models/Qwen3.5-4B")
    ap.add_argument("--advisee-backend", choices=["vllm", "openai"], default="vllm",
                    help="'openai' uses an API player (gpt-4o-mini / DeepSeek). Set "
                         "--advisee-model to the API model name.")
    ap.add_argument("--advisee-api-base", default=None,
                    help="OpenAI-compatible base_url (e.g. https://api.deepseek.com)")
    ap.add_argument("--advisee-api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--advisee-vision", action="store_true", default=False,
                    help="send player-view frame(s) to the API player (VL, e.g. gpt-4o-mini)")
    # Advisee (vllm player) sampling — defaults reproduce the old greedy behaviour.
    # Set per the player model's recommended params (e.g. Andy-4: T0.6 top_p0.95 top_k20).
    ap.add_argument("--advisee-temperature", type=float, default=0.0)
    ap.add_argument("--advisee-top-p", type=float, default=1.0)
    ap.add_argument("--advisee-top-k", type=int, default=-1)
    ap.add_argument("--advisee-min-p", type=float, default=0.0)
    ap.add_argument("--advisee-repetition-penalty", type=float, default=1.0)
    ap.add_argument("--verbose", action="store_true", default=False,
                    help="print per-step companion gate+advice and player action")
    ap.add_argument("--no-vision", action="store_true", default=False,
                    help="build MCEnv without vision (no frames in obs). Use for a "
                         "TEXT-ONLY companion (e.g. deepseek-chat) that cannot accept "
                         "images; VL companions (9B, gpt-4o-mini) should keep vision on.")
    ap.add_argument("--companion-url", default="http://localhost:8001")
    ap.add_argument("--companion-model", default="companion")
    ap.add_argument("--companion-backend", choices=["vllm", "openai"], default="vllm",
                    help="'openai' = API companion (e.g. gpt-4o-mini VL). Set "
                         "--companion-model to the API model name.")
    ap.add_argument("--companion-api-base", default=None)
    ap.add_argument("--companion-api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--adapter", default=None,
                    help="checkpoint to eval; hot-reloaded into the vLLM companion slot")
    ap.add_argument("--remap-lora", action="store_true", default=False)
    ap.add_argument("--mode", choices=["model-gate", "silence", "force-help", "fixed-step", "random-step"],
                    default="model-gate")
    ap.add_argument("--help-interval", type=int, default=0,
                    help="fixed-step mode only: silence steps BETWEEN forced HELPs "
                         "(0=every step, 1=every 2nd, 3=every 4th, 5=every 6th). "
                         "Advice still comes from the companion.")
    ap.add_argument("--tasks", default=",".join(CURRICULUM))
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    print("=" * 74)
    print(f"mc_eval  mode={args.mode}  adapter={args.adapter}")
    print("=" * 74, flush=True)

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # Hot-reload the eval adapter into the companion slot (vLLM companion only).
    if args.adapter and args.mode != "silence" and args.companion_backend == "vllm":
        path = args.adapter
        if args.remap_lora:
            path = remap_lora_for_vllm(args.adapter)
        ok = hot_reload_companion_lora(args.companion_url, args.companion_model, path)
        print(f"[eval] hot-reload adapter → {'ok' if ok else 'FAILED'}: {path}", flush=True)

    env = MCEnv(base_url=args.spark_url, max_steps=20, min_step_seconds=0.0,
                vision=not args.no_vision)
    if args.advisee_backend == "openai":
        from rl_causal.advisee import OpenAIAdvisee
        advisee = OpenAIAdvisee(model=args.advisee_model, base_url=args.advisee_api_base,
                                api_key_env=args.advisee_api_key_env, env="minecraft",
                                temperature=0.0, vision=args.advisee_vision)
        print(f"[advisee] API player: {args.advisee_model} "
              f"(base={args.advisee_api_base or 'openai'}, vision={args.advisee_vision})", flush=True)
    else:
        advisee = VllmAdvisee(model=args.advisee_model, url=args.advisee_url,
                              env="minecraft", temperature=args.advisee_temperature,
                              top_p=args.advisee_top_p, top_k=args.advisee_top_k,
                              min_p=args.advisee_min_p,
                              repetition_penalty=args.advisee_repetition_penalty,
                              vision=args.advisee_vision)
        print(f"[advisee] vllm player: {args.advisee_model} "
              f"(T={args.advisee_temperature} top_p={args.advisee_top_p} "
              f"top_k={args.advisee_top_k} min_p={args.advisee_min_p} "
              f"rep={args.advisee_repetition_penalty})", flush=True)
    companion = None
    if args.mode != "silence":
        if args.companion_backend == "openai":
            companion = OpenAICompanionMC(model=args.companion_model,
                                          base_url=args.companion_api_base,
                                          api_key_env=args.companion_api_key_env)
            print(f"[companion] API: {args.companion_model} "
                  f"(base={args.companion_api_base or 'openai'})", flush=True)
        else:
            companion = VllmCompanionRaw(model=args.companion_model, url=args.companion_url,
                                         env="minecraft")

    rng = random.Random(args.seed)
    results = []
    import time as _t
    for i in range(args.n_episodes):
        task = task_list[i % len(task_list)]   # round-robin for balanced coverage
        # Fault-tolerant: a transient MC-bot/tunnel drop must NOT abort the whole
        # eval cell. Retry the episode once after a short pause; on repeated
        # failure, record it as a fail (ms=0) and move on.
        r = None
        for _attempt in range(2):
            try:
                r = run_episode(env, advisee, companion, args.mode, task, seed=args.seed + i,
                                verbose=args.verbose, help_interval=args.help_interval)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  ep {i+1}/{args.n_episodes} [{task}] ERROR: "
                      f"{type(e).__name__}: {str(e)[:120]} (attempt {_attempt+1}/2)", flush=True)
                _t.sleep(8)
        if r is None:
            r = {"task": task, "won": False, "help": 0, "steps": 0, "ms_frac": 0.0}
        results.append(r)
        print(f"  ep {i+1}/{args.n_episodes} [{task}] "
              f"{'WIN ' if r['won'] else 'fail'} help={r['help']} steps={r['steps']} "
              f"ms={r['ms_frac']:.2f}", flush=True)

    # Aggregate
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r)
    tot_help = sum(r["help"] for r in results)
    tot_steps = sum(r["steps"] for r in results)
    n_won = sum(int(r["won"]) for r in results)

    def _hr(rs, won_only=False):
        """HELP rate = HELP-steps / steps over the episodes in rs. won_only
        restricts to successful episodes (advice efficiency on the wins)."""
        sub = [r for r in rs if r["won"]] if won_only else rs
        h = sum(r["help"] for r in sub)
        s = sum(r["steps"] for r in sub)
        return f"{h}/{s} ({100*h/max(1,s):.1f}%)" if s else "0/0 (n/a)"

    won_help = sum(r["help"] for r in results if r["won"])
    won_steps = sum(r["steps"] for r in results if r["won"])

    print("\n" + "=" * 74)
    print(f"MODE={args.mode}  n={len(results)}")
    print(f"  overall success : {n_won}/{len(results)} ({100*n_won/max(1,len(results)):.1f}%)")
    print(f"  HELP rate       : {tot_help}/{tot_steps} ({100*tot_help/max(1,tot_steps):.1f}%)")
    print(f"  HELP rate (win) : {won_help}/{won_steps} "
          f"({100*won_help/max(1,won_steps):.1f}%)")
    print(f"  milestone frac  : {sum(r['ms_frac'] for r in results)/max(1,len(results)):.2f}")
    print("  per-task  [ success | ms | HELP all | HELP win-only ]:")
    for task in task_list:
        rs = by_task.get(task, [])
        if rs:
            w = sum(int(r["won"]) for r in rs)
            print(f"    {task:24s} {w}/{len(rs)} ({100*w/len(rs):.0f}%)  "
                  f"ms={sum(r['ms_frac'] for r in rs)/len(rs):.2f}  "
                  f"HELP={_hr(rs)}  HELP(win)={_hr(rs, won_only=True)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
