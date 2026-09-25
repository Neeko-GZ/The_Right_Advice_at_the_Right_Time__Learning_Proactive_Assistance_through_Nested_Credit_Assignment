"""
coach.py — LIVE human-in-the-loop coach (MC Mode 2).

The HUMAN plays Minecraft; the trained companion watches and coaches, two-way:

  * PROACTIVE (gated): every `tick_seconds`, look at the player's state (inventory
    via the OP bot's /peek_inventory + optional screen frames), run the trained
    companion's gate+advice (build_companion_prompt from prompts.mc), and if the
    gate says HELP — and enough time has passed since the last hint — speak it
    into game chat via /say. This is the deployment of the NIC gate to a human.

  * REACTIVE (ungated): poll /poll_chat for what the human typed; answer each
    message directly (build_chat_reply_prompt from prompts.live) via /say.

ISOLATION: this file only IMPORTS companion prompts read-only. It never touches
mc_env.py, train_mc.py, the trainers, or the training contract. Delete rl_causal/
live/ + prompts/live.py + the three Node endpoints and training is untouched.

Observation sources
-------------------
  * inventory : GET  {spark_url}/peek_inventory?player=<name>   (OP bot /data get)
  * frames    : POST {this coach}:{frame_port}/frame  from the Windows streamer
                (base64 JPEG in the request body). Latest frame is used if fresh.
  * delivery  : POST {spark_url}/say {text}
  * human chat: GET  {spark_url}/poll_chat

Run:
    python -m rl_causal.live.coach \
        --player <player_name> \
        --task "mine iron and craft an iron pickaxe" \
        --milestones "collect wood,craft planks,craft sticks,craft table,craft wooden pickaxe,mine stone,craft stone pickaxe,mine iron,smelt iron,craft iron pickaxe" \
        --spark-url http://127.0.0.1:8765 \
        --companion-url http://localhost:8001 --companion-model companion
"""

from __future__ import annotations

import argparse
import os
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from rl_causal.prompts.mc import SYSTEM_PROMPT_COMPANION, build_companion_prompt  # noqa: F401 (eval-parity reference)
from rl_causal.prompts.mc_live import (
    SYSTEM_PROMPT_COMPANION_HUMAN, build_companion_prompt_human,
)
from rl_causal.prompts.live import (
    COACH_CHAT_SYSTEM_PROMPT, build_chat_reply_prompt,
    build_opening_next_step_prompt, build_milestone_celebration_prompt,
)


# ---------------------------------------------------------------------------
# tiny gate/advice parser (same shape the companion was trained to emit)
# ---------------------------------------------------------------------------

def _parse_gate_advice(raw: str):
    gm = re.search(r'"gate"\s*:\s*"([A-Za-z]+)"', raw or "")
    am = re.search(r'"advice"\s*:\s*"([^"]*)', raw or "")
    gate = gm.group(1).strip().upper() if gm else "SILENCE"
    advice = am.group(1).strip() if am else ""
    return gate, advice


# ---------------------------------------------------------------------------
# Final-craft ingredient requirements per human-study task (LIVE coach only).
# Used to tell the companion exactly what quantity is STILL MISSING for the
# final craft, so a small model does not claim "you can craft it" when the
# counts are too low. Each requirement: (label, matcher, need, same_variant).
#   matcher = ("suffix", "_wool")           -> any item name ending with it
#           = ("names", ("coal","charcoal")) -> these exact names
#   same_variant (suffix only): require ONE variant to reach `need` (a bed needs
#   3 wool of the SAME color); otherwise sum across variants.
# ---------------------------------------------------------------------------
_TASK_FINAL_REQS = {
    "make_bed": [
        ("wool of one color", ("suffix", "_wool"),   3, True),
        ("planks",            ("suffix", "_planks"), 3, False),
    ],
    "craft_stone_pickaxe": [
        ("cobblestone",    ("names", ("cobblestone",)),    3, False),
        ("sticks",         ("names", ("stick",)),          2, False),
        ("crafting table", ("names", ("crafting_table",)), 1, False),
    ],
    "obtain_iron_ingot": [
        ("furnace",     ("names", ("furnace",)),          1, False),
        ("raw iron",    ("names", ("raw_iron",)),         1, False),
        ("fuel (coal)", ("names", ("coal", "charcoal")),  1, False),
    ],
}


# Live-coach milestone tuning (per study task):
#   _MS_MIN_COUNT: a milestone needs at least this many of its target items
#     (default 1). e.g. a bed needs 3 wool + 3 planks, so "got wool"/"got planks"
#     should fire at 3, not 1.
#   _MS_CHECK_PLACED: these milestones also count if the block is PLACED nearby
#     (not in inventory) — a crafting table/furnace/bed is used by placing it on
#     the ground, so it may never sit in the hotbar.
_MS_MIN_COUNT = {
    "make_bed": {"got wool": 3, "got planks": 3},
    "craft_stone_pickaxe": {"got planks": 3, "got sticks": 2, "got stone": 3},
    "obtain_iron_ingot": {
        "got wood": 3, "got planks": 3, "got sticks": 2,
        "got stone": 3, "got coal": 1, "got raw iron": 1,
    },
}
_MS_CHECK_PLACED = {
    "make_bed": {"got table", "got bed"},
    "craft_stone_pickaxe": {"got table"},
    "make_furnace": {"got furnace"},
    "obtain_iron_ingot": {"got table", "got furnace"},
}


def _requirements_gap(task_name, counts):
    """Return a short 'still needs / now has enough' line for the final craft of
    a study task, or None when it's too early (player has none of the final
    ingredients yet) or the task has no requirement table. `counts` is name->qty."""
    reqs = _TASK_FINAL_REQS.get(task_name)
    if not reqs:
        return None
    have_any = False
    missing, ready = [], []
    for label, (kind, spec), need, same in reqs:
        if kind == "suffix":
            variants = [c for n, c in counts.items() if n.endswith(spec)]
            have = (max(variants) if variants else 0) if same else sum(variants)
        else:
            have = sum(counts.get(n, 0) for n in spec)
        if have > 0:
            have_any = True
        if have < need:
            missing.append(f"{need - have} more {label} (has {have}/{need})")
        else:
            ready.append(f"{label} ({have}/{need})")
    if not have_any:
        return None  # still gathering basics — don't fixate on the final craft yet
    # NOTE: phrased as reference-only. A crafting task ALWAYS has "missing"
    # ingredients until it is finished; if this line said "still needs X" plainly
    # the trained gate would read a permanent missing-prerequisite and fire HELP
    # every tick. The gate must decide from the stuck/idle signal, not from here.
    prefix = "Recipe check (for wording accuracy only, NOT a reason to speak): "
    if missing:
        line = prefix + "short " + "; ".join(missing) + "."
        if ready:
            line += " Enough: " + ", ".join(ready) + "."
        return line
    return (prefix + "the player now has enough for the final craft ("
            + ", ".join(ready) + ").")


# ---------------------------------------------------------------------------
# frame receiver — a tiny HTTP server the Windows streamer POSTs frames to
# ---------------------------------------------------------------------------

class _FrameStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._b64 = None
        self._ts = 0.0

    def put(self, b64: str):
        with self._lock:
            self._b64 = b64
            self._ts = time.time()

    def get_fresh(self, max_age_s: float = 5.0):
        with self._lock:
            if self._b64 and (time.time() - self._ts) <= max_age_s:
                return self._b64
            return None


def _start_frame_server(store: _FrameStore, port: int):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/frame":
                self.send_response(404); self.end_headers(); return
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(n) if n else b""
            try:
                b64 = body.decode("utf-8").strip()
                # accept either raw base64 or a data: URL
                if b64.startswith("data:"):
                    b64 = b64.split(",", 1)[-1]
                store.put(b64)
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
            except Exception as e:  # noqa: BLE001
                self.send_response(500); self.end_headers(); self.wfile.write(str(e).encode())

        def log_message(self, *a):  # silence per-request logging
            return

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[coach] frame receiver on :{port}/frame", flush=True)
    return srv


# ---------------------------------------------------------------------------
# the coach
# ---------------------------------------------------------------------------

class LiveCoach:
    def __init__(self, args):
        self.player = args.player
        self.task = args.task
        self.milestones = [m.strip() for m in (args.milestones or "").split(",") if m.strip()]
        self.spark_url = args.spark_url.rstrip("/")
        self.companion_url = args.companion_url.rstrip("/")
        self.companion_model = args.companion_model
        # optional bearer token, so the coach brain can be an API (DeepSeek /
        # OpenAI); omit for a local no-auth vLLM.
        self._auth = ""
        if getattr(args, "companion_api_key_env", None):
            self._auth = os.environ.get(args.companion_api_key_env, "").strip()
        self.tick_seconds = args.tick_seconds
        self.min_help_gap = args.min_help_gap
        self.temperature = args.temperature
        self.use_vision = not args.no_vision
        self.intro = getattr(args, "intro", "") or ""
        self.spectate = getattr(args, "spectate", False)
        self.bot_name = getattr(args, "bot_name", "spark") or "spark"

        self.frames = _FrameStore()
        if self.use_vision:
            _start_frame_server(self.frames, args.frame_port)

        self._last_help_ts = 0.0
        self._dialogue = []          # list of (speaker, text)
        self._last_inventory = []

        # ---- metric logging (eval-parity: ms / help / success / timing) ----
        self._task_name = getattr(args, "task_name", "") or ""
        self._ms = []
        if self._task_name:
            try:
                from rl_spark.task_pool import get_task_by_name
                self._ms = list(get_task_by_name(self._task_name).milestones)
            except Exception as e:
                print(f"[coach] could not load milestones for '{self._task_name}': {e}", flush=True)
        self._ms_total = max(1, len(self._ms))
        self._final_label = self._ms[-1].label if self._ms else None
        # if no explicit --milestones plan was given, use the task's milestone
        # labels as the companion's private plan (keeps plan == detection targets)
        if not self.milestones and self._ms:
            self.milestones = [m.label for m in self._ms]
        self._seen = set()               # fired milestone labels
        self._preseen = set()            # milestones already satisfied at session start
        self._ms_times = {}              # label -> seconds since start
        self._t0 = time.time()
        self._help_count = 0             # proactive HELP interventions
        self._decision_count = 0         # gate ticks actually evaluated
        self._useful_helps = 0           # HELP followed by a new milestone in-window
        self._pending_help_ts = None
        self._useful_window = getattr(args, "useful_window", 30.0)
        self._last_progress_ts = self._t0
        self._stuck_seconds = getattr(args, "stuck_seconds", 20.0)
        # --- pacing (A) + milestone-idle trigger (B) to fix bang-bang help-rate ---
        self._last_milestone_ts = self._t0        # for "no goal progress" trigger
        self._help_times = []                      # timestamps for per-minute cap
        self._recent_advice = []                   # (ts, advice) recently spoken
        self._milestone_stuck_seconds = getattr(args, "milestone_stuck_seconds", 40.0)
        self._repeat_window = getattr(args, "repeat_window", 90.0)
        self._max_helps_per_min = getattr(args, "max_helps_per_min", 3)
        self._debug_inv = getattr(args, "debug_inv", False)
        self._last_advice = ""    # last proactive hint spoken (anti-repeat guard)
        self._recipe_line = ""    # grounded final-craft gap (reactive channel only)
        self._tick_history = []   # rolling (action, result) per tick, so the
                                  # companion sees the recent trajectory (e.g. a
                                  # run of "no change" steps) and judges stuck itself
        # Proactive/gate channel: human-facing prompt that mirrors mc eval's
        # structure (same _state_block/_history_block), with only the deliberate
        # human adaptations (natural advice, no API/keyboard, missing-prerequisite
        # dropped from the gate trigger). The stuck signal is fed via obs["text"]'s
        # "Last action: idle Ns -> no change" line (trained format), so live_status
        # stays None.
        self._companion_system = SYSTEM_PROMPT_COMPANION_HUMAN
        self._won = False
        self._win_time = None
        self._condition = getattr(args, "condition", "") or ""
        self._log_path = getattr(args, "log", None) or f"/tmp/coach_{self.player}_{int(self._t0)}.json"

    # ---- Node bot I/O ----

    def _get_inventory(self):
        try:
            r = requests.get(f"{self.spark_url}/peek_inventory",
                             params={"player": self.player}, timeout=4)
            j = r.json() if r.ok else {}
            inv = j.get("inventory", []) if r.ok else []
            if getattr(self, "_debug_inv", False):
                who = j.get("player", self.player)
                print(f"[coach inv] player={who!r} -> "
                      + (", ".join(f"{it.get('name')}x{it.get('count')}" for it in inv) or "EMPTY"),
                      flush=True)
            return inv
        except Exception:
            return []

    def _game_command(self, command):
        """Run a raw OP command through the bot (spark /game_command)."""
        try:
            requests.post(f"{self.spark_url}/game_command",
                          json={"command": command}, timeout=4)
            print(f"[coach] game_command: {command}", flush=True)
        except Exception as e:
            print(f"[coach] game_command failed ({command}): {e}", flush=True)

    def _pull_frame(self):
        """Pull the latest bot-POV frame from spark GET /frames. Lets the coach
        run remotely (e.g. in-container) over the same spark link, with no push
        server or extra tunnel. Returns base64 JPEG or None."""
        try:
            r = requests.get(f"{self.spark_url}/frames", params={"n": 1}, timeout=4)
            if not r.ok:
                return None
            frames = r.json().get("frames", [])
            return frames[-1].get("jpeg_b64") if frames else None
        except Exception:
            return None

    def _say(self, text: str):
        if not text:
            return
        try:
            requests.post(f"{self.spark_url}/say", json={"text": text}, timeout=4)
            print(f"[coach→chat] {text}", flush=True)
        except Exception as e:
            print(f"[coach] /say failed: {e}", flush=True)

    def _poll_chat(self):
        try:
            r = requests.get(f"{self.spark_url}/poll_chat", timeout=4)
            return r.json().get("messages", []) if r.ok else []
        except Exception:
            return []

    # ---- observation ----

    def _pull_state(self):
        """Pull the bot-side compact scene (nearby blocks/entities, etc.). Reflects
        the PLAYER's surroundings only when the bot is co-located with them (e.g.
        --spectate); best-effort otherwise."""
        try:
            r = requests.get(f"{self.spark_url}/state_compact", timeout=4)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    def _build_obs(self):
        inv = self._get_inventory()          # the HUMAN's inventory (/peek_inventory)
        self._last_inventory = inv
        counts = {}
        for it in inv:
            counts[it.get("name", "")] = counts.get(it.get("name", ""), 0) + int(it.get("count", 1) or 1)
        names = []
        for n, c in counts.items():
            names += [n] * max(1, c)
        inv_text = ", ".join(f"{n} x{c}" for n, c in counts.items()) or "empty"

        # inventory delta since the last tick — a human has no discrete skill call,
        # so the change in items is what "just happened".
        prev = getattr(self, "_prev_inv_counts", None)
        now = time.time()
        bits = []
        if prev is not None:
            gained = [f"+{counts[n]-prev.get(n,0)} {n}" for n in counts if counts[n]-prev.get(n, 0) > 0]
            lost = [f"-{prev.get(n,0)-counts.get(n,0)} {n}" for n in prev if prev.get(n, 0)-counts.get(n, 0) > 0]
            bits = gained + lost
            if bits:
                self._last_progress_ts = now   # real progress resets the stuck timer
        self._prev_inv_counts = counts
        idle = now - getattr(self, "_last_progress_ts", now)
        # time since the last MILESTONE (not just any item pickup): lets the gate
        # fire when the human is busy but not actually advancing toward the goal.
        ms_idle = now - getattr(self, "_last_milestone_ts", now)

        # grounded final-craft gap — REACTIVE channel only (kept out of the gate obs
        # so a permanent "missing ingredient" line does not force HELP every tick).
        self._recipe_line = _requirements_gap(self._task_name, counts) or ""

        sc = self._pull_state()

        # Build obs["text"] in the SAME shape as eval's MCEnv.get_current_obs, so the
        # trained gate sees a familiar distribution instead of live-only line types.
        # The gate's main training signal is the "Last action: <act> -> <result>"
        # line; a human has none, so we synthesize it from the inventory delta
        # (progress) or an idle stretch (the stuck analog of an ineffective action).
        text_parts = []
        # ALWAYS report the player's inventory change since last tick (their
        # "what just happened"); bits includes losses, so crafting/consumption is
        # visible (e.g. "-4 planks, +1 crafting_table" = they just crafted a table).
        # When nothing changed, we report HOW LONG nothing has changed / no
        # milestone progress, and let the companion itself judge whether the
        # player is stuck — instead of the coach hard-thresholding it.
        if bits:
            text_parts.append(f"Last action: inventory change -> {', '.join(bits)}")
        else:
            text_parts.append(
                f"Last action: no inventory change for {int(idle)}s "
                f"(no goal progress for {int(ms_idle)}s)"
            )
        # record this tick into the rolling history (in eval's "action -> result"
        # shape) so the companion can see the recent trajectory and judge stuck.
        if bits:
            self._tick_history.append(("act", ", ".join(bits)))
        else:
            self._tick_history.append(("wait", f"no change ({int(idle)}s idle)"))
        self._tick_history = self._tick_history[-20:]
        # NOTE: sc["held_item"] is the SPARK BOT's hand, not the human player's,
        # so we do NOT report it (a spectator bot holding a stone_pickaxe made the
        # companion wrongly declare the task complete). The player's items come
        # only from /peek_inventory below.
        text_parts.append(f"Inventory: {inv_text}")
        nb = sc.get("nearby_blocks") or []
        self._last_nearby_names = {b.get("name") for b in nb if b.get("name")}
        if nb:
            text_parts.append("Nearby blocks: " + ", ".join(
                f"{b.get('name')} ({b.get('distance')}m {b.get('dir', '')})".replace(" )", ")").strip()
                for b in nb))
        nm = sc.get("nearby_top") or []
        if nm:
            text_parts.append("Nearby entities: " + ", ".join(
                f"{e.get('name')} ({e.get('distance')}m)" for e in nm))
        hp = sc.get("health"); fd = sc.get("food")
        if hp is not None or fd is not None:
            text_parts.append(f"Health: {hp}  Food: {fd}")

        obs = {
            "task_description": self.task,
            "text": "\n".join(text_parts),
            "inventory": names,
            "inventory_text": inv_text,
            "milestones_reference": self.milestones,
        }
        if self.use_vision:
            fr = self.frames.get_fresh() or self._pull_frame()
            if fr:
                obs["frames_b64"] = [fr]
        return obs

    # ---- metric tracking ----

    def _seed_baseline(self, inv_names):
        """Mark milestones already satisfied at session start as seen — WITHOUT
        celebrating them or counting them as reached this session. Prevents the
        setup kit (pre-given items) from triggering a spurious congrats at spawn
        and from inflating milestone metrics."""
        inv = set(inv_names or [])
        for m in self._ms:
            if inv & set(m.target_items):
                self._seen.add(m.label)
                self._preseen.add(m.label)
        if self._preseen:
            print(f"[coach] baseline (already satisfied at start, not counted): "
                  f"{sorted(self._preseen)}", flush=True)

    def _check_milestones(self, inv_names):
        """Detect milestone completions: quantity-aware, and counts blocks that are
        PLACED nearby (a crafting table / furnace / bed is used by placing it, so it
        may never sit in the inventory). On task completion, backfill any milestone
        that never fired and celebrate finishing."""
        counts = {}
        for n in (inv_names or []):
            counts[n] = counts.get(n, 0) + 1
        nearby = getattr(self, "_last_nearby_names", set()) or set()
        min_count = _MS_MIN_COUNT.get(self._task_name, {})
        placed_ok = _MS_CHECK_PLACED.get(self._task_name, set())
        now = time.time()
        newly = []
        for m in self._ms:
            if m.label in self._seen:
                continue
            need = min_count.get(m.label, 1)
            have = sum(counts.get(i, 0) for i in m.target_items)
            ok = have >= need
            if not ok and m.label in placed_ok and (nearby & set(m.target_items)):
                ok = True   # placed on the ground counts as done
            if not ok:
                continue
            self._seen.add(m.label)
            self._ms_times[m.label] = round(now - self._t0, 1)
            self._last_progress_ts = now
            self._last_milestone_ts = now
            newly.append(m.label)
            if self._pending_help_ts is not None and (now - self._pending_help_ts) <= self._useful_window:
                self._useful_helps += 1
                self._pending_help_ts = None
            if self._final_label and m.label == self._final_label and not self._won:
                self._won = True
                self._win_time = round(now - self._t0, 1)

        # auto-complete: once the task is won, backfill intermediate milestones that
        # never fired (e.g. table placed & missed, or bed crafted without a table).
        if self._won:
            for m in self._ms:
                if m.label not in self._seen:
                    self._seen.add(m.label)
                    self._ms_times.setdefault(m.label, round(now - self._t0, 1))
                    if m.label not in newly:
                        newly.append(m.label)

        if newly:
            print(f"[coach] milestone(s) reached: {newly}  ({len(self._seen)}/{self._ms_total})", flush=True)
            self._write_log()
            # non-API (ours) companion: force a warm line. On completion, one
            # "you finished" celebration (no next step); otherwise a per-subgoal cheer.
            if not self._auth:
                if self._won:
                    self._celebrate_milestone(self._final_label or newly[-1], completed=True)
                else:
                    self._celebrate_milestone(newly[-1])
        return newly

    def _celebrate_milestone(self, done_label, completed=False):
        """Ungated, forced congrats (non-API only). completed=True -> the whole
        task is done: congratulate finishing, no next-step nudge."""
        try:
            if completed:
                prompt = (f"The player has just COMPLETED the whole task: {self.task}. "
                          "In ONE short, warm sentence, congratulate them on finishing. "
                          "Do NOT suggest any further steps. Just the reply text, no JSON.")
            else:
                next_label = None
                for m in self._ms:
                    if m.label not in self._seen:
                        next_label = m.label
                        break
                prompt = build_milestone_celebration_prompt(self.task, done_label, next_label)
            line = self._call_companion(COACH_CHAT_SYSTEM_PROMPT, prompt,
                                        max_tokens=70, temperature=0.6)
            line = (line or "").strip().strip('"')
        except Exception as e:
            print(f"[coach] milestone celebration failed: {e}", flush=True)
            line = ""
        if not line:
            line = (f"You did it — {self.task} complete!" if completed
                    else f"Nice, you got {done_label.replace('_', ' ')}! Keep it up.")
        self._say(line)
        self._dialogue.append(("coach", line))
        self._last_help_ts = time.time()   # reset cooldown so we don't pile a hint on top

    def _write_log(self):
        data = {
            "player": self.player, "task": self.task, "task_name": self._task_name,
            "condition": self._condition, "companion_model": self.companion_model,
            "duration_s": round(time.time() - self._t0, 1),
            "won": self._won, "completion_time_s": self._win_time,
            "help_count": self._help_count, "decision_points": self._decision_count,
            "help_rate": round(self._help_count / self._decision_count, 3) if self._decision_count else None,
            "useful_helps": self._useful_helps,
            "post_help_progress": round(self._useful_helps / self._help_count, 3) if self._help_count else None,
            "milestones_total": self._ms_total,
            "milestones_preseen": len(self._preseen),
            "milestones_reached": len(self._seen - self._preseen),
            "ms_frac": round(len(self._seen - self._preseen)
                             / max(1, self._ms_total - len(self._preseen)), 3),
            "milestone_times_s": {k: v for k, v in self._ms_times.items()
                                  if k not in self._preseen},
        }
        try:
            with open(self._log_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[coach] log write failed: {e}", flush=True)
        return data

    # ---- companion call ----

    @staticmethod
    def _strip_think(s):
        """Remove <think>...</think> reasoning so it is never spoken to the player.
        Also handles a dangling closing tag (template opened think implicitly)."""
        s = s or ""
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
        if "</think>" in s:
            s = s.split("</think>")[-1]
        return s.strip()

    def _call_companion(self, system, user_content, max_tokens, temperature):
        payload = {
            "model": self.companion_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # local vLLM (no API key) -> disable thinking mode, matching training.
        # OpenAI would reject an unknown field, so only add it for the local path.
        if not self._auth:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Authorization": f"Bearer {self._auth}"} if self._auth else None
        r = requests.post(f"{self.companion_url}/v1/chat/completions",
                          json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return self._strip_think(r.json()["choices"][0]["message"]["content"])

    # ---- channels ----

    def _proactive_tick(self, obs):
        """Gated: run the trained companion; speak only if gate=HELP + pacing ok."""
        if (time.time() - self._last_help_ts) < self.min_help_gap:
            return  # still cooling down from the last hint — stay quiet
        self._decision_count += 1   # a gate decision is actually evaluated this tick
        user = build_companion_prompt_human(
            obs, history=self._tick_history[:-1], history_window=8,
        )  # companion sees recent trajectory and judges stuck itself
        try:
            raw = self._call_companion(self._companion_system, user,
                                       max_tokens=100, temperature=self.temperature)
        except Exception as e:
            print(f"[coach] companion(gate) failed: {e}", flush=True)
            return
        gate, advice = _parse_gate_advice(raw)
        print(f"[coach tick] gate={gate}" + (f" advice='{advice[:60]}'" if advice else "")
              + (f"  raw={raw[:200]!r}" if gate != "HELP" else ""),
              flush=True)
        if gate == "HELP" and advice:
            now = time.time()
            # pacing: hold ONLY if this hint repeats / paraphrases a RECENT spoken
            # hint (blocks "same subgoal, reworded" spam). A genuinely different
            # hint — e.g. the player just gathered enough and should now craft — is
            # NOT held, so the coach stays responsive to real state changes.
            self._recent_advice = [(t, a) for (t, a) in self._recent_advice
                                   if now - t < self._repeat_window]
            if any(self._too_similar(advice, a) for _, a in self._recent_advice):
                print("[coach tick] HELP held (repeat/paraphrase of a recent hint)", flush=True)
                self._last_help_ts = now
                return
            # hard per-minute cap
            self._help_times = [t for t in self._help_times if now - t < 60.0]
            if len(self._help_times) >= self._max_helps_per_min:
                print("[coach tick] HELP held (per-minute cap)", flush=True)
                self._last_help_ts = now
                return
            self._say(advice)
            self._dialogue.append(("coach", advice))
            self._last_advice = advice
            self._recent_advice.append((now, advice))
            self._last_help_ts = now
            self._help_times.append(now)
            self._help_count += 1
            self._pending_help_ts = self._last_help_ts

    @staticmethod
    def _too_similar(a, b, thresh=0.7):
        """True if two hints share most of their words (Jaccard over word sets)."""
        if not a or not b:
            return False
        sa = {w for w in re.findall(r"[a-z0-9]+", a.lower()) if len(w) > 2}
        sb = {w for w in re.findall(r"[a-z0-9]+", b.lower()) if len(w) > 2}
        if not sa or not sb:
            return False
        return len(sa & sb) / len(sa | sb) >= thresh

    def _handle_message(self, username, message, obs):
        """Ungated: the human asked something -> answer directly."""
        self._dialogue.append((username, message))
        # inject the grounded recipe gap here (reactive only) so a direct answer
        # respects exact quantities, without biasing the proactive gate.
        obs_r = obs
        if getattr(self, "_recipe_line", ""):
            obs_r = dict(obs)
            obs_r["text"] = self._recipe_line + "\n" + (obs.get("text", "") or "")
        user = build_chat_reply_prompt(obs_r, message, dialogue_history=self._dialogue)
        try:
            reply = self._call_companion(COACH_CHAT_SYSTEM_PROMPT, user,
                                         max_tokens=140, temperature=0.6)
        except Exception as e:
            print(f"[coach] companion(chat) failed: {e}", flush=True)
            return
        reply = reply.strip().strip('"')
        self._say(reply)
        self._dialogue.append(("coach", reply))
        # a reply counts as speaking — reset the proactive cooldown so we don't
        # immediately pile a hint on top of the answer.
        self._last_help_ts = time.time()

    # ---- main loop ----

    def run(self):
        print(f"[coach] player={self.player}  task='{self.task}'", flush=True)
        print(f"[coach] spark={self.spark_url}  companion={self.companion_url}"
              f"/{self.companion_model}", flush=True)
        print(f"[coach] tick={self.tick_seconds}s  min_help_gap={self.min_help_gap}s"
              f"  vision={self.use_vision}", flush=True)
        # wait for the bot to be reachable
        for _ in range(30):
            try:
                if requests.get(f"{self.spark_url}/health", timeout=3).ok:
                    break
            except Exception:
                pass
            time.sleep(2)

        # snapshot the starting inventory so pre-given setup items (planks, tools,
        # etc.) are not mistaken for milestones completed this session — otherwise
        # the coach congratulates the player at spawn and metrics are inflated.
        self._seed_baseline([it.get("name", "") for it in self._get_inventory()])

        # attach the spark bot to the player's point of view so the recorded
        # frames match what the participant sees (bot -> spectator, then spectate
        # the player). Requires OP/cheats. Verify the pulled frame actually
        # follows; if prismarine-viewer does not honor spectate, use --no-vision.
        if self.spectate:
            self._game_command(f"/gamemode spectator {self.bot_name}")
            time.sleep(0.5)
            self._game_command(f"/spectate {self.player} {self.bot_name}")
            time.sleep(0.5)

        # opening line so the participant sees the assistant is active (blind:
        # do not reveal which companion). Counts as speaking, so the first
        # proactive tick won't immediately pile a hint on top.
        if self.intro:
            line = self.intro.replace("{task}", self.task)
            self._say(line)
            self._dialogue.append(("coach", line))
            self._last_help_ts = time.time()
            # right after the greeting, hand the player a warm, concrete first step
            # so they are not left wondering where to begin (ungated opener).
            try:
                obs = self._build_obs()
                self._check_milestones(obs.get("inventory"))
                op = build_opening_next_step_prompt(self.task, obs)
                step = self._call_companion(COACH_CHAT_SYSTEM_PROMPT, op,
                                            max_tokens=80, temperature=0.6)
                step = (step or "").strip().strip('"')
                if step:
                    self._say(step)
                    self._dialogue.append(("coach", step))
                    self._last_help_ts = time.time()
            except Exception as e:
                print(f"[coach] opening next-step failed: {e}", flush=True)

        next_tick = time.time()
        try:
            while True:
                # 1) reactive: answer anything the human said (priority)
                msgs = self._poll_chat()
                if msgs:
                    obs = self._build_obs()
                    self._check_milestones(obs.get("inventory"))
                    for m in msgs:
                        self._handle_message(m.get("username", "player"),
                                             m.get("message", ""), obs)
                # 2) proactive: gated tick on schedule
                if time.time() >= next_tick:
                    obs = self._build_obs()
                    self._check_milestones(obs.get("inventory"))
                    self._proactive_tick(obs)
                    next_tick = time.time() + self.tick_seconds
                time.sleep(1.0)
        finally:
            self._write_log()
            print(f"[coach] session log -> {self._log_path}  "
                  f"(won={self._won} ms={len(self._seen)}/{self._ms_total} "
                  f"help={self._help_count}/{self._decision_count})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True, help="the human player's in-game name")
    ap.add_argument("--task", default="survive and progress in Minecraft")
    ap.add_argument("--milestones", default="",
                    help="comma-separated subgoal plan (only the coach sees this)")
    ap.add_argument("--spark-url", default="http://127.0.0.1:8765")
    ap.add_argument("--companion-url", default="http://localhost:8001")
    ap.add_argument("--companion-model", default="companion")
    ap.add_argument("--companion-api-key-env", default=None,
                    help="env var holding an API key (for DeepSeek/OpenAI as the "
                         "coach brain); omit for a local no-auth vLLM")
    ap.add_argument("--tick-seconds", type=float, default=8.0,
                    help="how often to run the proactive gate")
    ap.add_argument("--min-help-gap", type=float, default=15.0,
                    help="min seconds between spoken hints (anti-spam pacing)")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--frame-port", type=int, default=8770,
                    help="port this coach listens on for POST /frame from Windows")
    ap.add_argument("--no-vision", action="store_true",
                    help="inventory-only; ignore screen frames")
    ap.add_argument("--task-name", default="",
                    help="task_pool name (e.g. craft_stone_pickaxe) to load real "
                         "milestones for automatic ms/success detection + logging")
    ap.add_argument("--condition", default="",
                    help="condition label recorded in the session log (e.g. gpt-4o-mini / ours)")
    ap.add_argument("--log", default=None,
                    help="path to write the per-session JSON metrics log")
    ap.add_argument("--useful-window", type=float, default=30.0,
                    help="seconds: a HELP counts as 'useful' if a new milestone is "
                         "reached within this window after it")
    ap.add_argument("--stuck-seconds", type=float, default=20.0,
                    help="seconds of no inventory change after which the obs flags the "
                         "human as possibly idle (a human analog of the LLM player's "
                         "failed-action signal, so the gate can fire)")
    ap.add_argument("--milestone-stuck-seconds", type=float, default=40.0,
                    help="seconds of no MILESTONE progress after which the obs flags "
                         "'no goal progress' even if the player is picking up items "
                         "(fixes the 'busy but not advancing -> always silent' case)")
    ap.add_argument("--repeat-window", type=float, default=90.0,
                    help="a proactive hint is suppressed if it repeats/paraphrases any "
                         "hint spoken within this many seconds (blocks same-subgoal "
                         "spam); a genuinely different hint still goes through so the "
                         "coach stays responsive to state changes")
    ap.add_argument("--max-helps-per-min", type=int, default=3,
                    help="hard cap on proactive hints per 60s window")
    ap.add_argument("--spectate", action="store_true",
                    help="at start, put the spark bot into spectator mode and spectate "
                         "the player so recorded frames match the participant's POV")
    ap.add_argument("--bot-name", default="spark",
                    help="the spark bot's in-game name (for /spectate)")
    ap.add_argument("--debug-inv", action="store_true",
                    help="print the player name + inventory pulled each tick, to verify "
                         "the companion reads YOUR bag (not the bot's)")
    ap.add_argument("--intro",
                    default="Hi! I'm your assistant for this session. Your goal: {task}. "
                            "I'll chime in with a tip now and then. Good luck!",
                    help="one-time opening line at session start; '{task}' is replaced "
                         "by the task description (blind; set empty string to disable)")
    args = ap.parse_args()
    try:
        LiveCoach(args).run()
    except KeyboardInterrupt:
        print("\n[coach] stopped.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
