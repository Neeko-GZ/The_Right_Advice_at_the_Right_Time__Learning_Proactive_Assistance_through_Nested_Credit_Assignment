"""
gen_sft_teacher_v3.py — companion-voice advice + gate label via GPT-4o rewrite.

Version history:
  v3-alpha (deprecated): full plan → single 2-4 sentence prose response.
  v3-beta  (deprecated): staged tactical hints (1-2 next steps only), no gate.
  v3       (current):    STAGED advice + GATE decision {HELP, SILENCE}.

Motivation:
  Companion has two decisions per step:
    (1) GATE: whether to speak (HELP) or stay quiet (SILENCE)
    (2) CONTENT: if HELP, what tactical hint to give (1-3 sentences, staged)

  SFT provides warm-start for BOTH; PPO with DR credit assignment does the
  actual causal refinement of gate later (method.md § 5.7).

  Gate label heuristics (GPT-4o decides):
    - Player just followed expert plan correctly       → SILENCE
    - Player mid-execution of a subgoal (few steps in) → SILENCE
    - Player at start with no history                  → HELP (orient)
    - Player deviated from plan / stalled >2 steps     → HELP
    - Ambiguous exploration                            → SILENCE (default)

  Target rough split: 30-45% SILENCE, 55-70% HELP.

Input:  v2 JSONL (companion_teacher_v2.jsonl)
Output: v4 JSONL with fields:
  gate:            "HELP" | "SILENCE"
  advice:          str  (empty if SILENCE)
  advice_v2_backup: str (original full-plan advice preserved)

Design:
  - Stratify: for each task, keep 2 checkpoints (step 3 + step 7 by default)
    so ~3000 samples from ~1500 tasks.
  - Async concurrent calls (default 20) → 5-10 min for 3000.
  - Retry with backoff on rate limits; log failed samples.
  - Deterministic: same input JSONL + seed → same sampled subset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Companion-voice synthesis prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_REWRITE = (
    "You are helping create training data for a game companion assistant. "
    "The companion decides TWO things at each state:\n"
    "  (1) GATE — whether to SPEAK ('HELP') or stay QUIET ('SILENCE') this turn\n"
    "  (2) CONTENT — if HELP, WHAT to say (staged tactical advice, 1-3 sentences)\n\n"
    "Predict SILENCE when the player is doing fine and doesn't need interruption:\n"
    "  - Just took an action that matches the next step of the walkthrough\n"
    "  - Mid-execution of a subgoal (e.g. just picked up an object, going to place it)\n"
    "  - Reasonable exploration without significant deviation\n\n"
    "Predict HELP when the player would benefit from intervention:\n"
    "  - At the very start (no history, needs orientation)\n"
    "  - Deviated from the plan (visited wrong location, wrong object)\n"
    "  - Stalled — several actions with no progress\n"
    "  - Just completed a subgoal and needs the next pointer\n\n"
    "When HELP: advice must be STAGED. Give only the next 1-2 tactical actions, "
    "never the whole remaining walkthrough. The player will ask again later.\n\n"
    "Output STRICT JSON: {\"gate\": \"HELP\"|\"SILENCE\", \"advice\": \"...\"}. "
    "If SILENCE, advice may be an empty string."
)


def _build_rewrite_prompt(rec: dict) -> str:
    """Format the (task, history, current state, full plan) into a rewrite
    request. GPT must produce a companion-voice response."""
    task = rec.get("task_desc", "")
    state = rec.get("state_text", "")
    inv = rec.get("inventory_text", "") or "(nothing)"
    history = rec.get("history", []) or []
    full_plan = rec.get("advice", "")

    # Compact history rendering
    if history:
        h_lines = []
        for i, item in enumerate(history):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                act, obs = item
            else:
                act, obs = str(item), ""
            obs_short = (obs[:80] + "...") if len(obs) > 80 else obs
            h_lines.append(f"  step {i}: {act}   →   {obs_short}")
        h_block = "\n".join(h_lines)
    else:
        h_block = "  (no actions taken yet)"

    return (
        f"### Task\n{task}\n\n"
        f"### What the player has done so far\n{h_block}\n\n"
        f"### What the player currently sees\n{state}\n"
        f"Inventory: {inv}\n\n"
        f"### Full walkthrough (reference — helps you gauge progress)\n{full_plan}\n\n"
        f"### Instructions\n"
        f"Decide GATE and (if HELP) write advice. Requirements:\n\n"
        f"GATE selection (see system prompt for rules):\n"
        f"  - Aim for roughly 35-45% SILENCE overall\n"
        f"  - If uncertain, prefer HELP (missing a needed hint is worse than a "
        f"stray one)\n"
        f"  - No history at all → almost always HELP (orient the player)\n\n"
        f"CONTENT (only when HELP):\n"
        f"  - 1-3 sentences, flowing prose (NO numbered list, NO bullets)\n"
        f"  - Structure: brief recap → one-sentence diagnose → ONE concrete "
        f"immediate suggestion (very next action, or two tightly-coupled ones)\n"
        f"  - Do NOT dump the whole walkthrough. Later comes later.\n"
        f"  - Reference concrete objects/locations from task/state\n"
        f"  - Friendly but concise; avoid filler like 'Give it a try!'\n\n"
        f"### Output JSON:"
    )


# ---------------------------------------------------------------------------
# Sampling from v2 JSONL
# ---------------------------------------------------------------------------

def _load_and_sample(v2_path: str, per_task_ckpts: int,
                     target_ckpt_steps: tuple, seed: int) -> list:
    """Load v2 JSONL, group by task, pick checkpoints closest to target steps.

    v2 records include multiple checkpoints per task (steps 0, 3, 6, 10 by
    default). We pick per_task_ckpts of them per task, closest to target_ckpt_steps.
    """
    records = []
    with open(v2_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"[load] {len(records)} records from {v2_path}")

    # Group by (task_desc, gamefile) — task_desc alone may collide across gamefiles
    by_task = defaultdict(list)
    for r in records:
        key = r.get("task_desc", "") + "||" + str(r.get("meta", {}).get("gamefile", ""))
        by_task[key].append(r)

    def _step_of(r):
        # step count = len(history) as v2 records recorded checkpoint step
        return len(r.get("history", []))

    picked = []
    for _key, group in by_task.items():
        group.sort(key=_step_of)
        # For each target step, find closest in this group
        chosen = set()
        for target in target_ckpt_steps[:per_task_ckpts]:
            best = min(group, key=lambda r: abs(_step_of(r) - target))
            best_id = id(best)
            if best_id not in chosen:
                chosen.add(best_id)
                picked.append(best)

    rng = random.Random(seed)
    rng.shuffle(picked)
    print(f"[sample] picked {len(picked)} records "
          f"from {len(by_task)} tasks (~{per_task_ckpts}/task)")
    return picked


# ---------------------------------------------------------------------------
# OpenAI async client
# ---------------------------------------------------------------------------

import re as _re
_JSON_OBJ_RE = _re.compile(r"\{.*\}", _re.S)


def _parse_gate_advice(text: str) -> tuple:
    """Parse GPT output into (gate, advice). Robust to code fences and
    surrounding chatter. Returns (gate_str, advice_str) with gate normalized
    to 'HELP' or 'SILENCE' and advice possibly empty."""
    text = text.strip()
    # Strip common ```json fences
    if text.startswith("```"):
        text = _re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=_re.S).strip()

    obj = None
    # Try full parse first
    try:
        obj = json.loads(text)
    except Exception:
        m = _JSON_OBJ_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None

    if not isinstance(obj, dict):
        # Fallback: infer gate from text keywords; treat whole text as advice
        gate = "SILENCE" if _re.search(r"\bSILENCE\b", text, _re.I) else "HELP"
        advice = "" if gate == "SILENCE" else text
        return gate, advice, False   # ok_flag=False → schema violation

    gate_raw = str(obj.get("gate", "HELP")).strip().upper()
    gate = "SILENCE" if gate_raw.startswith("S") else "HELP"
    advice = str(obj.get("advice", "")).strip()
    if gate == "SILENCE":
        advice = ""   # normalize
    return gate, advice, True


async def _rewrite_one(client, model: str, rec: dict, max_retries: int = 3):
    """Rewrite one record → {gate, advice}. Returns (record, ok_flag, error_msg)."""
    prompt = _build_rewrite_prompt(rec)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_REWRITE},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=220,
                temperature=0.6,
                top_p=0.95,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            gate, advice, schema_ok = _parse_gate_advice(raw)
            out = dict(rec)
            out["advice_v2_backup"] = rec.get("advice", "")
            out["gate"] = gate
            out["advice"] = advice
            out["meta"] = dict(out.get("meta", {}))
            out["meta"]["v3_rewrite_model"] = model
            out["meta"]["v3_schema_ok"] = schema_ok
            return out, True, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(1.5 * (attempt + 1))
    return rec, False, last_err


async def _run_batch(records: list, model: str, concurrency: int,
                     out_path: str, progress_every: int = 50):
    """Concurrent rewrite of all records. Streaming write to out_path."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("ERROR: pip install openai  (async client required)")
        return 2

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in env")
        return 2

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    n_total = len(records)
    n_ok = 0
    n_fail = 0
    failed_ids = []
    t0 = time.time()
    lock = asyncio.Lock()

    async def _bound(idx, rec, f):
        nonlocal n_ok, n_fail
        async with sem:
            out_rec, ok, err = await _rewrite_one(client, model, rec)
            async with lock:
                if ok:
                    f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n_ok += 1
                else:
                    n_fail += 1
                    failed_ids.append((idx, err))
                if (n_ok + n_fail) % progress_every == 0:
                    dt = time.time() - t0
                    rate = (n_ok + n_fail) / max(1e-6, dt)
                    eta = (n_total - n_ok - n_fail) / max(1e-6, rate)
                    print(f"  [{n_ok+n_fail}/{n_total}]  ok={n_ok}  "
                          f"fail={n_fail}  rate={rate:.1f}/s  ETA={eta:.0f}s",
                          flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        tasks = [_bound(i, rec, f) for i, rec in enumerate(records)]
        await asyncio.gather(*tasks)

    dt = time.time() - t0
    print()
    print(f"[done] wrote {n_ok}/{n_total} to {out_path}  in {dt:.1f}s "
          f"({dt/60:.1f} min)")
    if n_fail:
        print(f"[warn] {n_fail} failed. First 5 errors:")
        for idx, err in failed_ids[:5]:
            print(f"  #{idx}: {err}")

    # Gate distribution + schema stats
    n_help = n_silence = n_schema_bad = 0
    n_help_no_hist = n_silence_with_hist = 0
    with open(out_path) as fr:
        for line in fr:
            try:
                r = json.loads(line)
            except Exception:
                continue
            g = r.get("gate", "?")
            hist_len = len(r.get("history", []) or [])
            if g == "HELP":
                n_help += 1
                if hist_len == 0: n_help_no_hist += 1
            elif g == "SILENCE":
                n_silence += 1
                if hist_len > 0: n_silence_with_hist += 1
            if not r.get("meta", {}).get("v3_schema_ok", True):
                n_schema_bad += 1
    total = n_help + n_silence
    if total > 0:
        print(f"[stats] gate distribution: "
              f"HELP={n_help} ({100*n_help/total:.1f}%)  "
              f"SILENCE={n_silence} ({100*n_silence/total:.1f}%)")
        print(f"        HELP w/ no history: {n_help_no_hist} "
              f"({100*n_help_no_hist/max(1,n_help):.1f}% of HELPs)")
        print(f"        SILENCE w/ history: {n_silence_with_hist} "
              f"({100*n_silence_with_hist/max(1,n_silence):.1f}% of SILENCEs)")
        print(f"        schema violations (fallback used): {n_schema_bad}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="v2 JSONL (e.g. /workspace/sft_data/companion_teacher_v2.jsonl)")
    ap.add_argument("--output", required=True,
                    help="v3 JSONL to write")
    ap.add_argument("--per-task-ckpts", type=int, default=2,
                    help="How many checkpoints to keep per task (default 2)")
    ap.add_argument("--target-steps", type=int, nargs="+", default=[3, 7, 0, 10],
                    help="Target step counts, closest one picked per task."
                         " Uses first --per-task-ckpts of these.")
    ap.add_argument("--model", default="gpt-4o",
                    help="OpenAI model for rewrite (default gpt-4o; "
                         "use gpt-4o-mini for cheap dryrun)")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap total samples (for dry-run)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    print("=" * 70)
    print("gen_sft_teacher_v3 — companion-voice rewrite")
    print("=" * 70)
    print(f"  input       : {args.input}")
    print(f"  output      : {args.output}")
    print(f"  per_task_ckpts : {args.per_task_ckpts}  target_steps={args.target_steps}")
    print(f"  model       : {args.model}   concurrency={args.concurrency}")
    if args.limit:
        print(f"  limit       : {args.limit}")
    print("=" * 70, flush=True)

    records = _load_and_sample(
        args.input,
        per_task_ckpts=args.per_task_ckpts,
        target_ckpt_steps=tuple(args.target_steps),
        seed=args.seed,
    )
    if args.limit:
        records = records[:args.limit]
        print(f"[limit] truncated to {len(records)}")

    return asyncio.run(_run_batch(
        records, args.model, args.concurrency, args.output
    ))


if __name__ == "__main__":
    sys.exit(main())
