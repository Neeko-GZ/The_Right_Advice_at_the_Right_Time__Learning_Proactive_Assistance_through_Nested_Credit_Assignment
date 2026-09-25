"""
mc_env.py — Minecraft environment for NIC (Nested Interventional Credit),
mirroring the interface of alfworld_env.py so the NIC loop
(cf_rollout_worker / cf_expander / cf_advantage / trainers) runs UNCHANGED.

Design decision (READ THIS):
  NIC requires a *replayable, discrete-step* env: at each HELP state it calls
  env.snapshot(), then env.restore(snap) + env.step(action) for each branch.
  This only works if the advisee is a **separate callable** whose command we
  execute one step at a time — exactly like ALFWorld.

  So in the RL loop the advisee is our own frozen Qwen3.5-4B (VllmAdvisee,
  called on vLLM :8000). It emits a Mindcraft skill command (e.g. "!collectBlocks('oak_log',4)");
  env.step(command) runs it via SparkAPI /execute_command, BYPASSING the bot's
  own LLM brain. The Mindcraft bot is the *body* (executes commands, streams
  frames + state); its andy.json brain is used only for the standalone
  "let 4B play" comm demo, not during NIC training.

Reward (mirrors alfworld_env.py dense-milestone design, §4.2 of the paper):
  Milestones come from the task definition (rl_spark/task_pool.py), detected by
  first appearance of target items in the bot INVENTORY (deterministic, no
  hallucination). Each milestone fires once and adds its coef; the terminal
  milestone (e.g. iron_pickaxe) fires 1.0 and ends the episode (success).

Snapshot / restore:
  Logical-state (NON bit-exact), per snapshot_restore_spec.md. Requires the
  Node-side SparkAPI to implement GET /snapshot + POST /restore (NOT YET DONE —
  see the SparkClientCF methods below, which are ready to call once Node lands
  them). Restore reconstructs _seen_milestones from the saved bookkeeping so
  branches do not re-fire subgoals already reached at the HELP state.

CRN caveat:
  MC world randomness (mob spawns, drops, ticks) is not fully controllable.
  set_seed() stores the CRN seed; the real reproducibility comes from the
  advisee's seedable sampling (VllmAdvisee seed, passed by cf_expander) + short
  N-step branches + a noise-floor threshold that filters high-variance nodes.
"""

from __future__ import annotations

import base64
import io
import os
import pickle
import random
import time
from typing import Any, Optional

from rl_causal.mindcraft_env import SparkClient
from rl_spark.task_pool import TaskSpec, get_task_by_name, sample_task


# ---------------------------------------------------------------------------
# SparkAPI client extended with snapshot / restore (Node endpoints per spec)
# ---------------------------------------------------------------------------

class SparkClientCF(SparkClient):
    """SparkClient + the two counterfactual endpoints from snapshot_restore_spec.md.

    NOTE: GET /snapshot and POST /restore must be implemented on the Node side
    (Mindcraft/mineflayer) before these work. Until then they raise on call.
    """

    def snapshot(self) -> dict:
        """Logical-state snapshot: position, yaw/pitch, inventory, vitals, time."""
        return self._get("/snapshot").json()["snapshot"]

    def restore(self, snap: dict, timeout: Optional[float] = None) -> dict:
        """Restore bot to a logical snapshot. Returns {ok, drift:{...}}."""
        return self._post("/restore", json={"snapshot": snap},
                          timeout=timeout if timeout is not None else max(self.timeout, 30.0)).json()

    def game_command(self, command: str) -> dict:
        """Run a raw in-game chat/cheat command (e.g. '/time set day')."""
        return self._post("/game_command", json={"command": command}).json()


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class MCEnv:
    """Minecraft env with the NIC (alfworld_env.py) interface.

    Public interface consumed by NIC:
      reset(seed=...)            -> (obs_dict, info)
      step(action)              -> (obs_dict, reward, done, truncated, info)
      snapshot()                -> bytes
      restore(snap: bytes)      -> None
      set_seed(crn)             -> None            (soft; see CRN caveat)
      rng_state()/set_rng_state()
      check_milestones(info)    -> list[str]
      current_milestones (prop) / milestone_detector (prop)
      admissible_commands (prop) / task_description (prop) / step_count (prop)
      get_current_obs()         -> dict
    Extra accessors for the VL policy:
      last_frames / last_state_compact / last_clip_text
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        task_name: Optional[str] = "obtain_iron_pickaxe",
        max_steps: int = 40,
        n_frames: int = 16,
        step_timeout: float = 60.0,
        top_inv: int = 12,
        top_near: int = 4,
        action_settle_s: float = 0.5,        # wait after a skill (which already blocks
                                             # until done) before reading state, so the
                                             # inventory/world tick reflects the action.
                                             # Applies to main + branches. Cheap insurance.
        min_step_seconds: float = 0.0,       # pace each step to >= this many seconds
                                             # (real-time UX; 0 for training speed).
                                             # We do NOT override the gate: that would
                                             # make gate_actual != model decision and
                                             # pollute the interventional credit.
        vision: bool = True,                 # attach frames to obs for a VL companion
        vision_n_frames: int = 1,            # how many latest frames to send
    ):
        self.client = SparkClientCF(base_url=base_url)
        self.task_name = task_name          # None → sample from pool each reset
        self.max_steps = max_steps
        self._default_max_steps = max_steps  # floor; per-episode budget = task.max_steps
        self.n_frames = n_frames
        self.step_timeout = step_timeout
        self.top_inv = top_inv
        self.top_near = top_near
        self.action_settle_s = action_settle_s
        self.min_step_seconds = min_step_seconds
        self.pacing_enabled = True   # paced on the main rollout; turn OFF during
                                     # branch expansion (branches are already slow)
        self.vision = vision
        self.vision_n_frames = vision_n_frames

        # Per-episode state
        self._task: Optional[TaskSpec] = None
        self._step_count = 0
        self._seen_milestones: set[str] = set()
        self._crn_seed: Optional[int] = None
        # Surface "home": captured once (first reset, at spawn) or from NIC_HOME
        # env ("x,y,z"). Every reset tp's the bot back here BEFORE setblock/summon,
        # so a previous episode's digDown can't leave the next one underground
        # (which would bury the spawned ores and put trees out of reach).
        self._home: Optional[tuple] = None
        _h = os.environ.get("NIC_HOME", "").strip()
        if _h:
            try:
                _xyz = [float(v) for v in _h.split(",")]
                if len(_xyz) == 3:
                    self._home = (round(_xyz[0]), round(_xyz[1]), round(_xyz[2]))
            except Exception:
                self._home = None

        # Cached last fetch (for VL policy + obs building)
        self._last_frames: list = []
        self._last_clip_text: Optional[str] = None
        self._last_state_compact: dict = {}

    # ------------------------ accessors for policy ---------------------

    @property
    def last_frames(self) -> list:
        return self._last_frames

    @property
    def last_clip_text(self) -> Optional[str]:
        return self._last_clip_text

    @property
    def last_state_compact(self) -> dict:
        return self._last_state_compact

    # ------------------------ inventory / milestones -------------------

    def _inventory_names(self, sc: Optional[dict] = None) -> set[str]:
        """Set of item ids currently in the bot inventory (+ held item),
        read from /state_compact. Milestone detection reads only this."""
        sc = sc if sc is not None else self._last_state_compact
        names: set[str] = set()
        for it in (sc.get("inventory_top") or sc.get("inventory") or []):
            n = it.get("name") if isinstance(it, dict) else None
            if n:
                names.add(n)
        held = sc.get("held_item")
        if isinstance(held, dict) and held.get("name"):
            names.add(held["name"])
        return names

    def _fire_milestones(self, inv_names: set[str]) -> tuple[float, int, bool]:
        """Fire (once) every task milestone whose target items are present.
        Returns (reward_delta, n_fired, terminal_reached).

        Uniform-coef dense reward: each milestone adds its coef; the task's
        FINAL milestone additionally ends the episode (success).
        """
        if self._task is None:
            return 0.0, 0, False
        reward = 0.0
        n_fired = 0
        terminal = False
        final_label = self._task.final_milestone().label
        for ms in self._task.milestones:
            if ms.label in self._seen_milestones:
                continue
            if inv_names & set(ms.target_items):
                self._seen_milestones.add(ms.label)
                reward += float(ms.coef)
                n_fired += 1
                if ms.label == final_label:
                    terminal = True
        # Terminal completion: reaching the goal implies the whole task is done,
        # so mark ALL milestones as seen (milestones aren't strictly progressive,
        # and transient/consumed intermediates can be missed by inventory snapshots).
        # Only affects the seen-set (ms_frac metric), NOT the dense reward above.
        if terminal:
            self._seen_milestones.update(m.label for m in self._task.milestones)
        return reward, n_fired, terminal

    def check_milestones(self, info: Optional[dict] = None) -> list[str]:
        """Return milestone labels currently satisfied but not yet seen, and
        mark them seen. (Parity with alfworld_env.check_milestones; the dense
        reward path in step() already handles firing, so this is auxiliary.)"""
        inv = self._inventory_names()
        fired = []
        if self._task is not None:
            for ms in self._task.milestones:
                if ms.label not in self._seen_milestones and (inv & set(ms.target_items)):
                    self._seen_milestones.add(ms.label)
                    fired.append(ms.label)
        return fired

    # ------------------------ gym-style API ----------------------------

    def reset(self, seed: Optional[int] = None,
              task_idx: Optional[int] = None,
              options: Optional[dict] = None) -> tuple[dict, dict]:
        if seed is not None:
            self._crn_seed = int(seed)
            random.seed(int(seed))
        # Pick task
        if self.task_name:
            self._task = get_task_by_name(self.task_name)
        else:
            self._task = sample_task()
        # Per-episode step budget from the task itself (harder tasks get more),
        # never below the configured floor.
        self.max_steps = max(self._default_max_steps,
                             int(getattr(self._task, "max_steps", 0) or 0))
        self._step_count = 0
        self._seen_milestones = set()
        self._last_action = ""
        self._last_action_result = ""

        self._wait_for_spawn()
        # Capture the surface home ONCE from the bot's position on the first reset.
        # DANGER: if the bot is currently underground (e.g. a previous run's digDown
        # left it in a deep hole), that underground spot becomes the home and EVERY
        # episode tp's back down there -> all tasks fail. So reject an obviously
        # subterranean capture (below _MIN_HOME_Y) and require an explicit NIC_HOME
        # surface coordinate instead. NIC_HOME (set in __init__) always wins.
        _MIN_HOME_Y = float(os.environ.get("NIC_HOME_MIN_Y", "60"))
        if self._home is None:
            try:
                _pos = (self.client.snapshot() or {}).get("position") or {}
                _x, _y, _z = _pos.get("x"), _pos.get("y"), _pos.get("z")
                if _x is not None and _y is not None and _z is not None:
                    if float(_y) < _MIN_HOME_Y:
                        print(f"[mc_env] WARNING: bot is at y={float(_y):.0f} (< {_MIN_HOME_Y:.0f}) "
                              f"— looks UNDERGROUND. Refusing to capture it as home (would tp every "
                              f"episode into a hole). Set NIC_HOME='x,y,z' to a surface spot, or move "
                              f"the bot above ground and rerun.", flush=True)
                    else:
                        self._home = (round(float(_x)), round(float(_y)), round(float(_z)))
                        print(f"[mc_env] home captured @ {self._home}", flush=True)
            except Exception:
                self._home = None
        # Reset the bot to a clean episode. We do NOT issue the task's
        # `!goal(...)` command: that would put the Mindcraft bot into its own
        # autonomous self-prompting loop, which conflicts with NIC's discrete
        # advisee-driven stepping (and would move the bot during snapshot/
        # restore). The task is conveyed to the advisee via its prompt
        # (task_text); the bot is only a body executing our per-step commands.
        _rk = dict(clear_inventory=True, restore_health=True, stop_actions=True)
        if self._home is not None:
            _rk.update(x=self._home[0], y=self._home[1], z=self._home[2])
        self.client.reset(**_rk, **(options or {}))
        # Let the tp land BEFORE setblock/summon so their `~` offsets resolve
        # against the home position, not wherever the bot happened to end up.
        if self._home is not None:
            time.sleep(0.6)
        # Per-task starting items (tools/prereqs) + summoned mobs, for interaction
        # tasks like milk_cow / shear_sheep. The bot runs these as OP; @s / ~ ~ ~
        # resolve to the bot itself. Given AFTER clear_inventory so they persist.
        for _it, _n in (getattr(self._task, "start_items", ()) or ()):
            try:
                self.client.game_command(f"/give @s minecraft:{_it} {int(_n)}")
            except Exception:
                pass
        for _mob, _n in (getattr(self._task, "spawn_mobs", ()) or ()):
            for _ in range(int(_n)):
                try:
                    self.client.game_command(f"/summon minecraft:{_mob} ~2 ~ ~1")
                except Exception:
                    pass
        # Fresh ore blocks placed near the bot (obtain_diamond etc.): distinct
        # offsets so they don't overwrite each other. New each episode -> the
        # world's natural ore is never consumed.
        _boff = [(2, 0, 0), (-2, 0, 1), (0, 0, 2), (1, 0, -2),
                 (3, 0, 1), (-3, 0, -1), (2, 0, 3), (-2, 0, -3)]
        _bi = 0
        for _blk, _n in (getattr(self._task, "spawn_blocks", ()) or ()):
            for _ in range(int(_n)):
                ox, oy, oz = _boff[_bi % len(_boff)]
                _bi += 1
                try:
                    self.client.game_command(f"/setblock ~{ox} ~{oy} ~{oz} minecraft:{_blk}")
                except Exception:
                    pass
        # Keep it permanently daytime so the VL companion's frames stay well-lit
        # (night darkens the view and degrades grounding). Disabling the daylight
        # cycle means snapshot/restore times also stay in the daytime range.
        try:
            self.client.game_command("/gamerule doDaylightCycle false")
            self.client.game_command("/time set day")
        except Exception:
            pass
        time.sleep(1.0)

        self._fetch_obs()
        # Seed _seen from whatever is already in inventory (usually empty).
        self._fire_milestones(self._inventory_names())
        info = {
            "task": self._task.name,
            "state_compact": self._last_state_compact,
            "clip_text": self._last_clip_text,
            "milestone_fired": False,
        }
        return self.get_current_obs(), info

    _LOG_TYPES = ("oak_log", "birch_log", "spruce_log", "jungle_log",
                  "acacia_log", "dark_oak_log", "mangrove_log", "cherry_log")

    def _normalize_wood_action(self, action: str) -> str:
        """Wood is species-agnostic in the tech tree (milestones accept ANY log /
        ANY planks; only the log->planks recipe is species-specific). Retarget:
          - collectBlocks("<log>"/"wood") -> nearest log actually nearby
          - craftRecipe("<X>_planks")     -> planks matching the log you HAVE
        so an oak-biased player still progresses with spruce/birch/etc."""
        import re
        a = action or ""
        # 1) collect wood -> nearest available log species
        m = re.match(r'\s*!collectBlocks\(\s*["\']?([a-z_]+)["\']?\s*,\s*(\d+)\s*\)', a)
        if m:
            want, n = m.group(1), m.group(2)
            if want in ("wood", "log", "logs") or want.endswith("_log"):
                nb = {b.get("name"): b.get("distance", 999)
                      for b in (self._last_state_compact.get("nearby_blocks") or [])}
                if want not in nb:
                    logs_near = [(d, name) for name, d in nb.items() if name in self._LOG_TYPES]
                    if logs_near:
                        return f'!collectBlocks("{min(logs_near)[1]}", {n})'
            return a
        # 2) craft planks -> match the log species you actually hold
        m2 = re.match(r'\s*!craftRecipe\(\s*["\']?([a-z_]+)_planks["\']?\s*,\s*(\d+)\s*\)', a)
        if m2:
            want_wood, n = m2.group(1), m2.group(2)
            inv = set(self._inventory_names(self._last_state_compact))
            if f"{want_wood}_log" not in inv:
                have = [lt for lt in self._LOG_TYPES if lt in inv]
                if have:
                    return f'!craftRecipe("{have[0].replace("_log", "")}_planks", {n})'
            return a
        # 3) craft bed -> match the wool color you actually hold (bed = <color>_bed,
        #    needs 3 same-color wool). Retarget "bed"/wrong-color -> a color you have.
        m3 = re.match(r'\s*!craftRecipe\(\s*["\']?([a-z_]*bed)["\']?\s*,\s*(\d+)\s*\)', a)
        if m3:
            want_bed, n = m3.group(1), m3.group(2)
            counts = {it.get("name"): it.get("count", 0)
                      for it in (self._last_state_compact.get("inventory_top") or [])}
            wool = sorted(((c, name) for name, c in counts.items()
                           if name.endswith("_wool") and c >= 3), reverse=True)
            if wool:
                have_color = wool[0][1].replace("_wool", "")
                req_color = want_bed[:-4] if want_bed.endswith("_bed") else ""
                keep = bool(req_color) and counts.get(f"{req_color}_wool", 0) >= 3
                if not keep:
                    return f'!craftRecipe("{have_color}_bed", {n})'
        return a

    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        """Execute one advisee command (Mindcraft skill) via /execute_command,
        then read state and compute the dense milestone reward.

        `action` is the advisee's chosen command string (e.g. "!collectBlocks('oak_log',4)").
        Returns (obs_dict, reward, done, truncated, info) — gym-style 5-tuple.
        """
        _t0 = time.time()
        action = self._normalize_wood_action(action)
        self._last_action = action or ""
        self._last_action_result = ""
        if action and action.strip():
            try:
                _res = self.client.execute_command(action, timeout=self.step_timeout)
                _r = _res.get("result") if isinstance(_res, dict) else _res
                # Surface the skill's status string ("Collected 5 oak_log" /
                # "Could not find any oak_log to collect") so the player can tell
                # success from a no-op and adapt instead of looping.
                self._last_action_result = str(_r).strip() if _r else "(no visible effect)"
            except Exception:
                self._last_action_result = "(action failed)"
            if self.action_settle_s > 0:
                time.sleep(self.action_settle_s)   # let the world tick reflect the action

        self._fetch_obs()
        self._step_count += 1

        # Real-time pacing (UX only): pad the step to at least min_step_seconds.
        # Does NOT touch the gate, so gate_actual == the model's decision and the
        # interventional credit stays clean. Keep 0 during training for speed.
        if self.min_step_seconds > 0 and self.pacing_enabled:
            _elapsed = time.time() - _t0
            if _elapsed < self.min_step_seconds:
                time.sleep(self.min_step_seconds - _elapsed)

        inv = self._inventory_names()
        reward, n_fired, terminal = self._fire_milestones(inv)

        sc = self._last_state_compact
        dead = (sc.get("health") or 0) <= 0
        done = bool(terminal or dead)
        truncated = self._step_count >= self.max_steps

        info = {
            "task": self._task.name if self._task else None,
            "state_compact": sc,
            "clip_text": self._last_clip_text,
            "milestone_fired": n_fired > 0,     # help-cost recency reset (mirrors ALFWorld)
            "won": terminal,
            "dead": dead,
            "step": self._step_count,
        }
        return self.get_current_obs(), float(reward), done, bool(truncated), info

    def close(self):
        pass

    # ------------------------ snapshot / restore -----------------------

    def snapshot(self) -> bytes:
        """Opaque blob (bytes, like ALFWorld). Wraps the SparkAPI logical
        snapshot + our episode bookkeeping so restore is self-contained.

        Requires Node-side GET /snapshot (snapshot_restore_spec.md §2).
        """
        spark_snap = self.client.snapshot()
        state = {
            "spark": spark_snap,
            "task_name": self._task.name if self._task else None,
            "step_count": self._step_count,
            "seen_milestones": set(self._seen_milestones),
            "crn_seed": self._crn_seed,
            "rng_state": self.rng_state(),
            "last_state_compact": dict(self._last_state_compact),
            "last_clip_text": self._last_clip_text,
        }
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, snap: bytes) -> None:
        """Restore bot logical state + episode bookkeeping to the snapshot
        instant. Reconstructs _seen_milestones so branches don't re-fire
        subgoals already reached at the HELP state (mirrors alfworld_env
        rebuilding the detector history).

        Requires Node-side POST /restore (snapshot_restore_spec.md §3).
        """
        state = pickle.loads(snap)
        # Push logical state back into the world (tp + inventory + time + vitals).
        self.client.restore(state["spark"])

        # Restore bookkeeping to the snapshot instant.
        if state.get("task_name"):
            self._task = get_task_by_name(state["task_name"])
        self._step_count = int(state["step_count"])
        self._seen_milestones = set(state["seen_milestones"])
        self._crn_seed = state.get("crn_seed")
        self.set_rng_state(state["rng_state"])
        self._last_state_compact = dict(state.get("last_state_compact", {}))
        self._last_clip_text = state.get("last_clip_text")

        # Safety: reconcile _seen with the actually-restored inventory. Items
        # present after restore imply their milestones are already reached; we
        # keep the saved _seen (authoritative) but also mark any present-item
        # milestone as seen, so a slightly-imperfect restore can't cause a
        # spurious re-fire on the first branch step.
        self._fetch_obs()
        inv = self._inventory_names()
        if self._task is not None:
            for ms in self._task.milestones:
                if inv & set(ms.target_items):
                    self._seen_milestones.add(ms.label)

    # ------------------------ CRN / RNG --------------------------------

    def set_seed(self, crn_seed: int) -> None:
        """Store the CRN seed. MC world randomness is not fully controllable;
        the effective CRN comes from the advisee's seedable sampling (the
        expander passes the seed to VllmAdvisee) + short branches + a
        noise-floor threshold. See the CRN caveat in the module docstring."""
        self._crn_seed = int(crn_seed)
        random.seed(int(crn_seed))

    def rng_state(self) -> bytes:
        py_state = random.getstate()
        try:
            import numpy as np
            np_state = np.random.get_state()
        except ImportError:
            np_state = None
        return pickle.dumps({"py": py_state, "np": np_state})

    def set_rng_state(self, state: bytes) -> None:
        d = pickle.loads(state)
        random.setstate(d["py"])
        if d.get("np") is not None:
            try:
                import numpy as np
                np.random.set_state(d["np"])
            except ImportError:
                pass

    # ------------------------ properties -------------------------------

    @property
    def current_milestones(self) -> list:
        return list(self._task.milestones) if self._task else []

    @property
    def milestone_detector(self):
        """Parity with alfworld_env; MC bakes milestone firing into step(),
        so downstream reward.py does not need a separate detector."""
        return self

    @property
    def admissible_commands(self) -> list[str]:
        """MC advisee emits free-form skill commands, so there is no fixed
        admissible set. Returned empty; the advisee prompt lists the skill
        vocabulary instead."""
        return []

    @property
    def task_description(self) -> str:
        return self._task.task_text if self._task else ""

    @property
    def step_count(self) -> int:
        return self._step_count

    # ------------------------ obs building -----------------------------

    def get_current_obs(self) -> dict:
        """obs dict for the current state (for prompt building + branch use).
        Text summary from /state_compact + clip_text; frames are read by the
        VL policy via .last_frames."""
        sc = self._last_state_compact
        inv = sorted(self._inventory_names(sc))
        held = (sc.get("held_item") or {}).get("name")
        ms_ref = [f"{m.label}: {'/'.join(m.target_items[:3])}"
                  for m in (self._task.milestones if self._task else [])]
        text_parts = []
        if self._last_clip_text:
            text_parts.append(f"Scene: {self._last_clip_text}")
        la = getattr(self, "_last_action_result", "")
        if la:
            text_parts.append(f"Last action: {getattr(self, '_last_action', '')} -> {la[:160]}")
        _inv_top = sc.get("inventory_top", []) or []
        _inv_str = (", ".join(f"{it.get('name')} x{it.get('count', 1)}" for it in _inv_top)
                    if _inv_top else "empty")
        text_parts.append(f"Held: {held or 'nothing'}")
        text_parts.append(f"Inventory: {_inv_str}")
        nb = sc.get("nearby_blocks", []) or []
        if nb:
            nb_str = ", ".join(
                f"{b['name']} ({b['distance']}m {b.get('dir', '')})".replace(" )", ")").strip()
                for b in nb
            )
            text_parts.append(f"Nearby blocks: {nb_str}")
        nm = sc.get("nearby_top", []) or []
        if nm:
            nm_str = ", ".join(f"{e.get('name')} ({e.get('distance')}m)" for e in nm)
            text_parts.append(f"Nearby entities: {nm_str}")
        text_parts.append(f"Health: {sc.get('health')}  Food: {sc.get('food')}")
        obs = {
            "text": "\n".join(text_parts),
            "task_description": self.task_description,
            "admissible_commands": [],
            "inventory": inv,
            "held_item": held,
            "health": sc.get("health"),
            "food": sc.get("food"),
            "nearby": sc.get("nearby", []),
            "nearby_blocks": nb,
            "step_count": self._step_count,
            "milestones_reference": ms_ref,
        }
        # Vision: attach the latest frame(s) as base64 JPEG for a VL companion.
        # The text advisee ignores this field; only the companion prompt uses it.
        if self.vision and self._last_frames:
            obs["frames_b64"] = self._encode_frames(self._last_frames[-self.vision_n_frames:])
        return obs

    @staticmethod
    def _encode_frames(frames: list) -> list:
        """PIL images -> base64 JPEG strings (no data: prefix)."""
        out = []
        for im in frames:
            try:
                buf = io.BytesIO()
                im.convert("RGB").save(buf, format="JPEG", quality=80)
                out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            except Exception:
                pass
        return out

    # ------------------------ helpers ----------------------------------

    def _fetch_obs(self) -> None:
        """Refresh cached frames + state_compact from SparkAPI."""
        try:
            fr = self.client.frames(n=self.n_frames)
            self._last_frames = fr.frames
            self._last_clip_text = fr.clip_text
        except Exception:
            self._last_frames = []
        try:
            self._last_state_compact = self.client.state_compact(
                top_inv=self.top_inv, top_near=self.top_near)
        except Exception:
            self._last_state_compact = {}

    def _wait_for_spawn(self, timeout: float = 30.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                h = self.client.health()
                if h.get("spawned") and h.get("frame_recorder"):
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError("Bot did not spawn within timeout; check SparkAPI + Mindcraft.")
