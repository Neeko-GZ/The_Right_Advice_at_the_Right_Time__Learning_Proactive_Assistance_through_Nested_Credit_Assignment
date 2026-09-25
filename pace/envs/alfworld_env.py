"""
alfworld_env.py — ALFWorld environment wrapper for rl_causal.

Wraps AlfredTWEnv (TextWorld backend) with:
  - reset(task_idx, seed)
  - step(action)
  - snapshot() / restore() via pickle           (ALFWorld == "exact" case in
                                                 method.md § 5.4)
  - rng_state() / set_rng_state()               (for CRN across branches)
  - check_milestones(info) → list[str]          (subgoal detection)
  - admissible_commands / task_description properties

Design notes:
  - ALFWorld is fully serializable via pickle (Python-native TextWorld state)
    → snapshot/restore is exact; PC3 fidelity noise floor should be ~0 (float
    roundoff only).
  - RNG spans Python random + numpy random; both captured on snapshot.
  - The advisee LLM's context is NOT captured here — the rollout loop
    handles that separately (see brancher.py contract in docs/interfaces.md).
  - For unit tests without alfworld installed, use MockALFWorldEnv which
    implements the same interface with a deterministic mini-task.
"""

from __future__ import annotations

import hashlib
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# obs schema — parity principle
# ---------------------------------------------------------------------------
#
# Design rule ("对等原则"): the COMPANION must not receive any perceptual
# information the ADVISEE would not have. Otherwise "did advice help?" is
# confounded by an information advantage — reviewers will ask.
#
# So:
#   CompanionObs = (frame, task_description, history)     ← same across envs
#   AdviseeObs   = above + env-specific action interface  ← env-specific
#
# `ALFObs` remains as the FULL container carried by the env internally
# (includes env-specific fields like inventory_text, admissible_commands,
# gamefile). It is projected down to CompanionObs / AdviseeObs before
# handing off to policies.
#
# Milestone detection reads whatever the researcher has access to (env info
# dict, inventory, or a VLM judge on the frame) — that's privileged
# training-time signal, neither agent sees it.
# ---------------------------------------------------------------------------

@dataclass
class CompanionObs:
    """What the companion sees. Uniform across ALFWorld / MC / Gaming-500."""
    frame:            Any            # np.ndarray HxWx3 RGB; None if not yet supported
    task_description: str            # natural-language goal
    history:          list[str] = field(default_factory=list)
    # bookkeeping (not perceptual)
    step_count: int = 0

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,                # kept as-is; caller decides serialize
            "task_description": self.task_description,
            "history": list(self.history),
            "step_count": self.step_count,
        }

    def hash(self) -> str:
        """Deterministic hash for grouping / dedup. Does not include the
        raw frame — for that use a downstream perceptual hash."""
        h = hashlib.sha1()
        h.update(self.task_description.encode("utf-8", errors="ignore"))
        h.update(b"|")
        for step in self.history[-3:]:  # recent history only
            h.update(step.encode("utf-8", errors="ignore"))
            h.update(b"\n")
        return h.hexdigest()[:16]


@dataclass
class AdviseeObs:
    """What the advisee sees. Same as CompanionObs plus env-specific fields
    (action interface + advice from companion, if any). Each env wrapper
    fills the env-specific slots differently."""
    # ── shared with companion ────────────────────────────────────
    frame:            Any
    task_description: str
    history:          list[str] = field(default_factory=list)
    step_count: int = 0
    # ── env-specific ─────────────────────────────────────────────
    admissible_commands: list[str] = field(default_factory=list)  # ALFWorld native
    text_obs:            str = ""                                  # e.g. ALFWorld text state
    inventory_text:      str = ""                                  # MC / ALFWorld
    # ── advice from companion (None if SILENCE) ───────────────────
    advice: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "task_description": self.task_description,
            "history": list(self.history),
            "step_count": self.step_count,
            "admissible_commands": list(self.admissible_commands),
            "text_obs": self.text_obs,
            "inventory_text": self.inventory_text,
            "advice": self.advice,
        }


@dataclass
class ALFObs:
    """FULL observation container carried inside the env wrapper.

    Neither companion nor advisee sees this directly — use
    `to_companion_obs()` / `to_advisee_obs()` to project down. Kept because
    the wrapper internally needs env-specific bookkeeping (task_type,
    step_count, etc.).
    """
    text: str                    # human-readable observation (env-native text)
    task_description: str        # what the agent is trying to do
    frame:                Any = None   # image (H×W×3 RGB) or None if env doesn't render
    admissible_commands: list[str] = field(default_factory=list)
    inventory_text: str = ""     # "You are carrying: nothing." etc.
    history:              list[str] = field(default_factory=list)
    step_count: int = 0
    task_type: str = ""          # e.g. "pick_and_place_simple"

    # ── projections down to what each agent sees ────────────────

    def to_companion_obs(self) -> CompanionObs:
        return CompanionObs(
            frame=self.frame,
            task_description=self.task_description,
            history=list(self.history),
            step_count=self.step_count,
        )

    def to_advisee_obs(self, advice: Optional[str] = None) -> AdviseeObs:
        return AdviseeObs(
            frame=self.frame,
            task_description=self.task_description,
            history=list(self.history),
            step_count=self.step_count,
            admissible_commands=list(self.admissible_commands),
            text_obs=self.text,
            inventory_text=self.inventory_text,
            advice=advice,
        )

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "task_description": self.task_description,
            "frame": self.frame,
            "admissible_commands": list(self.admissible_commands),
            "inventory_text": self.inventory_text,
            "history": list(self.history),
            "step_count": self.step_count,
            "task_type": self.task_type,
        }

    def hash(self) -> str:
        """Deterministic hash for grouping (GiGPO-style)."""
        h = hashlib.sha1()
        h.update(self.text.encode("utf-8", errors="ignore"))
        h.update(b"|")
        h.update(self.task_description.encode("utf-8", errors="ignore"))
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _unbatched(x):
    """ALFWorld returns single-batch fields as lists like ['foo']; unwrap."""
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    return x


def _extract_task_type(info: dict) -> str:
    """Parse the task-type from ALFWorld's `extra.gamefile` path.

    Path format:
      .../train/<TASK_TYPE>-<obj>-<mod>-<recep>-<id>/<trial>/game.tw-pddl

    Returns the leading task_type portion (e.g. 'pick_two_obj_and_place').
    """
    gamefile = _unbatched(info.get("extra.gamefile", "")) if info else ""
    if not gamefile:
        return ""
    parts = str(gamefile).split("/")
    if len(parts) < 3:
        return ""
    # trial dir is parts[-2]; the task-config dir is parts[-3]
    task_dir = parts[-3]
    # first token before `-` is the task type
    return task_dir.split("-", 1)[0]


def _extract_task_description(obs_text: str, info: dict) -> str:
    """Return the natural-language task instruction.

    ALFWorld embeds it in obs after 'Your task is to:'. Prefer parsing
    obs text; fall back to task-type identifier.
    """
    marker = "Your task is to:"
    if isinstance(obs_text, str) and marker in obs_text:
        after = obs_text.split(marker, 1)[1].strip()
        # Task instruction is usually a single sentence ending in '.'
        end = after.find(".")
        if end > 0:
            return after[:end + 1].strip()
        return after.split("\n", 1)[0].strip()
    return _extract_task_type(info)


_MILESTONE_VERB_TO_ENGLISH = {
    "picked_up": "Pick up",
    "placed": "Place",
    "cleaned": "Clean",
    "heated": "Heat",
    "cooled": "Cool",
    "sliced": "Slice",
    "toggled": "Activate",
}


def _format_milestones_readable(milestones: list) -> list[str]:
    """Convert ALFTrajPlanDetector milestone labels into human-readable
    reference steps for the companion prompt (asymmetric info).

    Milestone.label format: '<verb>_<obj_type>_<semantic_idx>' e.g.
      'picked_up_winebottle_0' → 'Pick up the winebottle'
      'cooled_winebottle_1'   → 'Cool the winebottle'
      'placed_winebottle_2'   → 'Place the winebottle'
      'task_complete'         → skipped (terminal marker, not a step)

    Verbs that don't parse are left as-is so nothing crashes; the
    milestone is still shown as "<verb> <obj>" which is still usable.
    """
    out: list[str] = []
    for m in milestones:
        label = getattr(m, "label", str(m))
        if label == "task_complete":
            continue
        # Split off trailing semantic_idx
        parts = label.rsplit("_", 1)
        core = parts[0] if len(parts) == 2 and parts[1].isdigit() else label
        # Split core into verb + object type
        # verb keys can contain underscores ("picked_up"), so match longest prefix
        verb_key = None
        for vk in _MILESTONE_VERB_TO_ENGLISH:
            if core.startswith(vk + "_") or core == vk:
                verb_key = vk
                break
        if verb_key is None:
            out.append(core.replace("_", " "))
            continue
        obj = core[len(verb_key):].lstrip("_")
        verb_en = _MILESTONE_VERB_TO_ENGLISH[verb_key]
        if obj:
            out.append(f"{verb_en} the {obj}")
        else:
            out.append(verb_en)
    return out


def _expand_env_vars(obj):
    """Recursively expand `$VAR` and `${VAR}` in string values of a nested
    dict/list. Non-string leaves are passed through untouched.

    ALFWorld's base_config.yaml uses `$ALFWORLD_DATA/json_2.1.1/train`
    style paths; we expand them in-place so alfworld doesn't receive
    literal `$…` strings that break `os.path.exists`.
    """
    import os
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


# ---------------------------------------------------------------------------
# real ALFWorld wrapper (lazy import)
# ---------------------------------------------------------------------------

class ALFWorldEnv:
    """ALFWorld env wrapper implementing the SnapshottableEnv contract.

    Requires `alfworld` package installed. Import is deferred so this file
    can be imported (and MockALFWorldEnv tested) without alfworld present.
    """

    # Task types with unreliable expert traces (W28-EXP-011 finding):
    # `pick_two_obj_and_place` had 0/3 expert-completion rate on our
    # sampling. Excluded so SFT warm-start data quality stays high.
    # Override at construction time via `excluded_task_types` (pass set()
    # to disable filter entirely; useful for eval / ablation).
    EXCLUDED_TASK_TYPES: set[str] = {"pick_two_obj_and_place"}

    def __init__(
        self,
        config_path: Optional[str] = None,
        train_eval: str = "train",       # "train" | "eval_in_distribution" | "eval_out_of_distribution"
        batch_size: int = 1,
        seed: int = 0,
        excluded_task_types: Optional[set[str]] = None,
    ) -> None:
        # Lazy import — fail loudly with a helpful message if alfworld missing
        try:
            import alfworld  # noqa: F401
            import alfworld.agents.environment as alf_env
        except ImportError as e:
            raise ImportError(
                "alfworld not installed. Install with `pip install alfworld "
                "textworld` and run `alfworld-download` to get task data."
            ) from e

        # Load ALFWorld config (yaml). If not given, look at env var or default.
        if config_path is None:
            import os
            config_path = os.environ.get("ALFWORLD_CONFIG")
            if config_path is None:
                raise ValueError(
                    "Pass config_path or set ALFWORLD_CONFIG env var."
                )

        # Import here to avoid namespace pollution at module level
        import yaml
        with Path(config_path).open("r") as f:
            self.config = yaml.safe_load(f)

        # ALFWorld configs reference `$ALFWORLD_DATA` in path fields.
        # Expand shell-style env vars in-place so alfworld doesn't get
        # a literal `$ALFWORLD_DATA/...` string. ALFWORLD_DATA MUST be
        # set in the process environment before construction.
        if "ALFWORLD_DATA" not in __import__("os").environ:
            raise ValueError(
                "ALFWORLD_DATA env var not set. Point it at the directory "
                "holding json_2.1.1/, logic/, and detectors/ (run "
                "`alfworld-download` first if not downloaded)."
            )
        self.config = _expand_env_vars(self.config)

        # Build env via ALFWorld's factory. Newer alfworld versions
        # (0.4+) no longer expose AlfredTWEnv as a top-level attribute;
        # you must use get_environment("AlfredTWEnv") to resolve it.
        if not hasattr(alf_env, "get_environment"):
            raise RuntimeError(
                "alfworld.agents.environment.get_environment not found — "
                "unexpected alfworld API. Check version."
            )
        # Optional override: force the split for ALL constructions in this
        # process via NIC_TRAIN_EVAL (e.g. train the companion directly on
        # valid_seen as an oracle/upper-bound probe). Unset -> use the arg.
        _te_override = __import__("os").environ.get("NIC_TRAIN_EVAL", "").strip()
        if _te_override:
            train_eval = _te_override
            print(f"[env] NIC_TRAIN_EVAL override -> train_eval={train_eval}",
                  flush=True)

        env_cls = alf_env.get_environment("AlfredTWEnv")
        # AlfredTWEnv is a factory: constructing it loads game files,
        # then init_env(batch_size) RETURNS the actual TextWorld env
        # (a gym-like object with reset/step). Store BOTH:
        #   _factory  keeps the game-list state (which game is next)
        #   _env      is what we call reset/step on
        self._factory = env_cls(self.config, train_eval=train_eval)
        self._env = self._factory.init_env(batch_size=batch_size)

        # Track step count locally (some ALFWorld versions don't expose it)
        self._step_count = 0
        # Cache the last obs / info so admissible_commands is queryable
        self._last_obs_text: str = ""
        self._last_info: dict = {}
        # Task-type + description cache (populated on reset)
        self._task_type: str = ""
        self._task_desc: str = ""
        # Action history for replay-based snapshot/restore. ALFWorld's
        # underlying TextWorld runtime uses ctypes CDLL, which cannot
        # be pickled — snapshot only records what to replay.
        self._action_history: list[str] = []
        # Milestones auto-loaded per task from traj_data.json (list of
        # Milestone objects); populated by reset(). Empty until reset().
        self._current_milestones: list = []
        # Detector for the ALFWorld traj-plan milestones. Kept as a
        # member so reset()/step() can talk to the same instance and
        # the rollout loop can access it via `env.milestone_detector`.
        from rl_causal.reward import ALFTrajPlanDetector
        self._traj_detector = ALFTrajPlanDetector()
        # Milestones already fired this episode (persisted across step)
        self._seen_milestones: set[str] = set()
        # Task-type filter — None → use class default; empty set disables.
        self._excluded_task_types: set[str] = (
            set(self.EXCLUDED_TASK_TYPES) if excluded_task_types is None
            else set(excluded_task_types)
        )
        # Store seed for reproducibility diagnostics
        self._seed: int = int(seed)
        random.seed(self._seed)
        try:
            import numpy as np
            np.random.seed(self._seed)
        except ImportError:
            pass

    # ── lifecycle ──────────────────────────────────────────────────

    def reset(self, task_idx: Optional[int] = None,
              seed: Optional[int] = None,
              max_task_retries: int = 20) -> tuple[dict, dict]:
        """Reset to a new task; return (obs_dict, info_dict).

        `task_idx` selects which task (if the env cycles through a task
        list); leave None to advance to the next task. `seed` reseeds
        Python/numpy RNGs for reproducibility.

        If the underlying env yields a task in `self._excluded_task_types`
        (e.g. `pick_two_obj_and_place`, W28-EXP-011), reset() re-advances
        up to `max_task_retries` times to skip past it. Empty exclusion
        set disables this behaviour.
        """
        if seed is not None:
            self._seed = int(seed)
            random.seed(self._seed)
            try:
                import numpy as np
                np.random.seed(self._seed)
            except ImportError:
                pass

        # Retry loop: keep resetting until we hit a non-excluded task
        # type. On single-batch env each reset advances the game iterator
        # by one, so retries naturally sample different tasks.
        obs, info, text, task_type = None, {}, "", ""
        # Enough retries to skip both excluded task types AND occasional
        # malformed-PDDL gamefiles (see below).
        _tries = max(25, max(1, max_task_retries))
        _last_err = None
        for _ in range(_tries):
            # Some ALFWorld gamefiles ship malformed PDDL that the TextWorld
            # parser (tatsu) cannot parse (FailedToken / EOF errors). Rather
            # than let the exception bubble up and force the worker to rebuild
            # the whole AlfredTWEnv (~11s each), catch it here and skip to the
            # next game — reset() advances the game iterator on each call.
            try:
                obs, info = self._env.reset()
            except Exception as e:  # noqa: BLE001 — textworld/tatsu parse errors
                _last_err = e
                continue
            text = obs[0] if isinstance(obs, (list, tuple)) else str(obs)
            info = info if isinstance(info, dict) else {}
            task_type = _extract_task_type(info)
            if task_type not in self._excluded_task_types:
                break
        else:
            raise RuntimeError(
                f"reset: no valid task in {_tries} retries "
                f"(excluded={self._excluded_task_types}; last_parse_err={_last_err!r}). "
                f"Check ALFWorld base_config.yaml task_types, or pass "
                f"excluded_task_types=set() to disable the filter."
            )

        self._step_count = 0
        self._last_obs_text = text
        self._last_info = info
        self._seen_milestones = set()
        self._action_history = []
        self._task_type = task_type
        self._task_desc = _extract_task_description(text, info)

        # Load milestones for this task instance from traj_data.json.
        # gamefile lives at <trial>/game.tw-pddl; traj_data.json is
        # next to it. If not found (e.g. custom task), fall back to
        # an empty milestone list — downstream reward will only get
        # the terminal task_complete signal.
        gamefile = info.get("extra.gamefile")
        gamefile = _unbatched(gamefile) if gamefile else ""
        # Persist the gamefile for snapshot(): step() info does NOT carry
        # extra.gamefile, so reading _last_info after any warmup step yields
        # None (W32 fix — that None silently became load(None) once restore
        # actually started using the target gamefile).
        self._current_gamefile = str(gamefile) if gamefile else None
        try:
            self._current_milestones = self._traj_detector.load_traj(str(gamefile))
        except Exception as e:
            # Non-fatal — log via placeholder; env still usable
            self._current_milestones = []

        # Cache human-readable form for companion prompt (asymmetric info).
        # Advisee obs will NOT include this field.
        self._milestones_readable = _format_milestones_readable(self._current_milestones)

        obs_struct = self._build_obs(text, info)
        obs_dict = obs_struct.to_dict()
        # Attach milestones as a top-level obs field — build_companion_prompt
        # reads it, build_advisee_prompt intentionally ignores it.
        obs_dict["milestones_reference"] = list(self._milestones_readable)
        return obs_dict, info

    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        """Take an action (one of the admissible commands).

        Returns (obs_dict, reward, done, truncated, info) — gym-style.

        The `reward` returned here is the ENV's native reward (usually
        0/1 on success). Our downstream `assemble_reward` in reward.py
        computes the full v2 reward from this + milestone detection.
        """
        obs, rewards, dones, info = self._env.step([action])
        text = obs[0] if isinstance(obs, (list, tuple)) else str(obs)
        reward = float(rewards[0]) if isinstance(rewards, (list, tuple)) else float(rewards)
        done = bool(dones[0]) if isinstance(dones, (list, tuple)) else bool(dones)
        info = info if isinstance(info, dict) else {}

        self._step_count += 1
        self._last_obs_text = text
        self._last_info = info
        self._action_history.append(action)
        # Notify the traj-plan detector so subsequent milestone.check()
        # calls see the up-to-date action history.
        self._traj_detector.observe_action(action)

        # ── Dense subtask reward (W33) ──────────────────────────────────
        # Fire each fine-grained plan milestone (picked_up / placed / cleaned
        # / heated / cooled / sliced / toggled) at most once per episode, and
        # add its coef ON TOP of the raw env reward. Terminal success already
        # lives in the raw env `reward` (env gives ~1 on win), so we do NOT
        # re-add task_complete here — only the intermediate milestones, to
        # densify the counterfactual signal (was: sparse 0/1 terminal only).
        # Fire-once uses self._seen_milestones, which snapshot()/restore()
        # already persist, so branches only fire milestones NOT yet reached
        # at the HELP state.
        _n_ms_fired = 0
        for _ms in self._current_milestones:
            if _ms.label in self._seen_milestones:
                continue
            # Skip terminal task_complete — success is already in the raw env
            # `reward` (~1 on win); firing it here would double-count. Only the
            # intermediate plan milestones densify the signal.
            if getattr(_ms, "label", "") == "task_complete":
                continue
            try:
                if self._traj_detector.check(info, _ms):
                    self._seen_milestones.add(_ms.label)
                    reward += float(getattr(_ms, "coef", 0.0))
                    _n_ms_fired += 1
            except Exception:
                pass
        # Flag milestone completion this step so the help-cost recency counter
        # can be reset (a fresh subtask reached → immediate next help is NOT
        # "consecutive spam", so its penalty is forgiven).
        if isinstance(info, dict):
            info["milestone_fired"] = _n_ms_fired > 0

        obs_struct = self._build_obs(text, info)
        obs_dict = obs_struct.to_dict()
        # Carry milestones through step() so companion sees them at every step
        obs_dict["milestones_reference"] = list(getattr(self, "_milestones_readable", []))
        # ALFWorld doesn't distinguish done vs truncated; we treat all as done.
        return obs_dict, reward, done, False, info

    # ── snapshot / restore (exact via pickle) ──────────────────────

    def snapshot(self) -> bytes:
        """Snapshot via action-replay metadata (not env pickle).

        ALFWorld's TextWorld runtime holds `SyncBatchEnv` which wraps
        `ctypes.CDLL` (native Inform 7 interpreter) — this cannot be
        pickled. Instead we save enough to REPLAY: the current gamefile
        + the action history since the last reset. `restore()` re-seeds,
        re-resets, iterates to the same gamefile, and replays actions.

        Trade-off: restore is O(n_actions × ~10ms per action) instead of
        O(1) pickle load. For typical brancher usage (≤ 30 steps),
        restore costs ~300ms — acceptable.

        method.md § 5.4's "ALFWorld = exact case" is preserved in the
        SEMANTIC sense: replay deterministically reproduces state.
        """
        # Extract current gamefile so restore can target it. Prefer the value
        # cached at reset() — step() info drops extra.gamefile, so _last_info
        # would give None after any step.
        gf = getattr(self, "_current_gamefile", None)
        if not gf:
            gf = self._last_info.get("extra.gamefile") if self._last_info else None
            if isinstance(gf, (list, tuple)) and gf:
                gf = gf[0]

        state = {
            "gamefile": gf,
            "action_history": list(self._action_history),
            "step_count": self._step_count,
            "last_obs_text": self._last_obs_text,
            "last_info": dict(self._last_info) if self._last_info else {},
            "task_type": self._task_type,
            "task_desc": self._task_desc,
            "seen_milestones": set(self._seen_milestones),
            "seed": self._seed,
            "rng_state": self.rng_state(),
        }
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, snap: bytes) -> None:
        """Restore by action replay: reset → find target game → step
        through recorded actions.

        Postcondition: env internal state matches snapshot instant, and
        any subsequent step(a) yields the same (obs, r, done) as it did
        at snapshot time (given shared RNG via set_rng_state).
        """
        state = pickle.loads(snap)
        target_gamefile = state["gamefile"]

        # Reset RNG BEFORE env reset so game selection is deterministic
        self.set_rng_state(state["rng_state"])
        self._seed = int(state["seed"])
        random.seed(self._seed)
        try:
            import numpy as np
            np.random.seed(self._seed)
        except ImportError:
            pass

        # Direct-hop to target game. CRITICAL (W32 root-cause fix):
        # TextworldBatchGymEnv.reset() pulls gamefiles from the GYM ENV's own
        # `self._gamefiles_iterator` (a shuffled cycle over the FULL pool),
        # built at init_env() time. The previous code patched the *factory*'s
        # game_files/_gamefiles_iterator — a DIFFERENT object that reset()
        # never reads — so every restore landed on a random game (different
        # object layout, non-deterministic). That desynced every CF branch and
        # made the counterfactual signal identically zero. Fix: pin the GYM
        # ENV's iterator to the target gamefile, reset, then restore the
        # original iterator so normal (non-restore) rollouts are unaffected.
        import itertools
        gym_env = self._env
        orig_iter = getattr(gym_env, "_gamefiles_iterator", None)
        if orig_iter is not None:
            try:
                gym_env._gamefiles_iterator = itertools.repeat(target_gamefile)
                obs, info = gym_env.reset()
            finally:
                gym_env._gamefiles_iterator = orig_iter
        else:
            # Fallback: sequential reset until match (slow, but works even if
            # the gym env's internal structure differs across textworld versions)
            MAX_TRIES = 5000
            for _ in range(MAX_TRIES):
                obs, info = self._env.reset()
                cur_gf = info.get("extra.gamefile") if isinstance(info, dict) else None
                if isinstance(cur_gf, (list, tuple)) and cur_gf:
                    cur_gf = cur_gf[0]
                if cur_gf == target_gamefile:
                    break
            else:
                raise RuntimeError(
                    f"restore: could not locate target gamefile after "
                    f"{MAX_TRIES} resets: {target_gamefile}"
                )

        # Replay recorded actions
        _dbg = bool(__import__("os").environ.get("BRANCH_DEBUG"))
        _last_replay_obs = None
        for action in state["action_history"]:
            _o, _r, _d, _i = self._env.step([action])
            _last_replay_obs = _o
            if _dbg:
                _ot = _o[0] if isinstance(_o, (list, tuple)) else _o
                print(f"      [restore-replay] act={action!r} -> {str(_ot)[:90]!r}",
                      flush=True)
        if _dbg:
            _rt = _last_replay_obs[0] if isinstance(_last_replay_obs, (list, tuple)) else _last_replay_obs
            print(f"      [restore-check] post-replay REAL obs = {str(_rt)[:90]!r}", flush=True)
            print(f"      [restore-check] snapshot CACHED obs = {str(state.get('last_obs_text',''))[:90]!r}",
                  flush=True)

        # Restore bookkeeping to snapshot instant
        self._step_count = int(state["step_count"])
        self._last_obs_text = state["last_obs_text"]
        self._last_info = state["last_info"]
        self._task_type = state["task_type"]
        self._task_desc = state["task_desc"]
        self._current_gamefile = target_gamefile
        self._seen_milestones = set(state["seen_milestones"])
        self._action_history = list(state["action_history"])
        # Rebuild the traj detector's internal action history so fine-milestone
        # check() is correct post-restore: replay above used the RAW _env.step
        # and never called observe_action, so the detector would otherwise hold
        # the FULL main-trajectory history (stale) instead of the HELP-state one.
        self._traj_detector.reset_history()
        for _a in state["action_history"]:
            self._traj_detector.observe_action(_a)
        # Re-apply RNG state after replay (replay may have advanced it)
        self.set_rng_state(state["rng_state"])

    # ── RNG state (for CRN across branches) ────────────────────────

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

    # ── milestones ─────────────────────────────────────────────────

    def check_milestones(self, info: Optional[dict] = None) -> list[str]:
        """Return newly-fired milestone labels this step.

        Implementation: ALFWorld tasks have `won` flag on success; we
        also parse the current observation for progress markers (picked
        up, cleaned, heated, etc.). For W28 the coarse "task_complete"
        milestone is sufficient; finer milestones can be added in W29-W30
        as we build a task-specific decomposition matching task_pool.py.
        """
        info = info or self._last_info
        fired: list[str] = []

        if info.get("won", False) and "task_complete" not in self._seen_milestones:
            fired.append("task_complete")
            self._seen_milestones.add("task_complete")

        return fired

    # ── properties ─────────────────────────────────────────────────

    @property
    def current_milestones(self) -> list:
        """Task-instance milestones (from traj_data.json, W28-EXP-022).
        Empty until reset() succeeds."""
        return list(self._current_milestones)

    @property
    def milestone_detector(self):
        """The ALFTrajPlanDetector wired to this env — pass to reward_milestone."""
        return self._traj_detector

    @property
    def admissible_commands(self) -> list[str]:
        cmds = self._last_info.get("admissible_commands")
        if isinstance(cmds, list) and cmds and isinstance(cmds[0], list):
            return list(cmds[0])
        return list(cmds) if isinstance(cmds, list) else []

    @property
    def task_description(self) -> str:
        return getattr(self, "_task_desc", "") or self._task_type

    @property
    def step_count(self) -> int:
        return self._step_count

    # ── helpers ────────────────────────────────────────────────────

    def get_current_obs(self) -> dict:
        """Return obs dict for the current env state (for brancher use).
        Reads from cached _last_obs_text / _last_info populated by last
        reset() or step()."""
        return self._build_obs(self._last_obs_text, self._last_info).to_dict()

    def _build_obs(self, text: str, info: dict) -> ALFObs:
        cmds = info.get("admissible_commands")
        if isinstance(cmds, list) and cmds and isinstance(cmds[0], list):
            cmds = cmds[0]
        elif not isinstance(cmds, list):
            cmds = []

        return ALFObs(
            text=text,
            task_description=getattr(self, "_task_desc", "") or self._task_type,
            admissible_commands=list(cmds),
            inventory_text=info.get("inventory", ""),
            step_count=self._step_count,
            task_type=self._task_type,
        )


# ---------------------------------------------------------------------------
# Mock ALFWorld — deterministic mini-env for unit tests
# ---------------------------------------------------------------------------

class MockALFWorldEnv:
    """A tiny deterministic env that mimics ALFWorld's interface without
    needing alfworld installed.

    Task: "pick up the mug and put it in the drawer."
    Milestones:
      - "picked_up_mug"      when action == "take mug"
      - "task_complete"      when action == "put mug in drawer" (after M1)

    Admissible commands change based on state:
      initial:      [look, take mug, open drawer]
      has_mug:      [look, put mug in drawer]
      opened:       [look, take mug, close drawer]
      has_mug+open: [look, put mug in drawer]

    Snapshot/restore via pickle of internal dict — trivial roundtrip.
    """

    ACTIONS_INITIAL = ["look", "take mug", "open drawer"]

    def __init__(self, seed: int = 0) -> None:
        self._seed = int(seed)
        random.seed(self._seed)
        self._state = self._initial_state()
        self._step_count = 0
        self._seen_milestones: set[str] = set()
        self._done = False
        self._won = False

    def _initial_state(self) -> dict:
        return {"has_mug": False, "drawer_open": False, "mug_placed": False}

    # ── lifecycle ──────────────────────────────────────────────────

    def reset(self, task_idx: Optional[int] = None,
              seed: Optional[int] = None) -> tuple[dict, dict]:
        if seed is not None:
            self._seed = int(seed)
            random.seed(self._seed)
        self._state = self._initial_state()
        self._step_count = 0
        self._seen_milestones = set()
        self._done = False
        self._won = False

        obs = ALFObs(
            text="You are in a kitchen. There is a mug on the counter and a closed drawer.",
            task_description="put the mug in the drawer",
            admissible_commands=self._compute_admissible(),
            inventory_text="You are carrying: nothing.",
            step_count=0,
            task_type="mock_pick_and_place",
        )
        info = {"won": False, "admissible_commands": [obs.admissible_commands]}
        return obs.to_dict(), info

    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        if self._done:
            # Terminal — no-op
            return self._build_obs_dict(), 0.0, True, False, {"won": self._won}

        reward = 0.0
        if action == "look":
            pass  # no-op except step count bump
        elif action == "take mug" and not self._state["has_mug"]:
            self._state["has_mug"] = True
        elif action == "open drawer" and not self._state["drawer_open"]:
            self._state["drawer_open"] = True
        elif action == "close drawer" and self._state["drawer_open"]:
            self._state["drawer_open"] = False
        elif action == "put mug in drawer" and self._state["has_mug"] and self._state["drawer_open"]:
            self._state["mug_placed"] = True
            self._state["has_mug"] = False
            self._done = True
            self._won = True
            reward = 1.0
        # else: invalid action, no state change

        self._step_count += 1
        info = {
            "won": self._won,
            "admissible_commands": [self._compute_admissible()],
        }
        return self._build_obs_dict(), reward, self._done, False, info

    # ── snapshot / restore ─────────────────────────────────────────

    def snapshot(self) -> bytes:
        return pickle.dumps({
            "state": dict(self._state),
            "step_count": self._step_count,
            "seen_milestones": set(self._seen_milestones),
            "seed": self._seed,
            "done": self._done,
            "won": self._won,
            "rng_state": self.rng_state(),
        })

    def restore(self, snap: bytes) -> None:
        d = pickle.loads(snap)
        self._state = dict(d["state"])
        self._step_count = int(d["step_count"])
        self._seen_milestones = set(d["seen_milestones"])
        self._seed = int(d["seed"])
        self._done = bool(d["done"])
        self._won = bool(d["won"])
        self.set_rng_state(d["rng_state"])

    def rng_state(self) -> bytes:
        return pickle.dumps({"py": random.getstate()})

    def set_rng_state(self, state: bytes) -> None:
        d = pickle.loads(state)
        random.setstate(d["py"])

    # ── milestones ─────────────────────────────────────────────────

    def check_milestones(self, info: Optional[dict] = None) -> list[str]:
        fired: list[str] = []
        if self._state["has_mug"] and "picked_up_mug" not in self._seen_milestones:
            fired.append("picked_up_mug")
            self._seen_milestones.add("picked_up_mug")
        if self._state["mug_placed"] and "task_complete" not in self._seen_milestones:
            fired.append("task_complete")
            self._seen_milestones.add("task_complete")
        return fired

    # ── properties ─────────────────────────────────────────────────

    @property
    def admissible_commands(self) -> list[str]:
        return self._compute_admissible()

    @property
    def task_description(self) -> str:
        return "put the mug in the drawer"

    @property
    def step_count(self) -> int:
        return self._step_count

    def _compute_admissible(self) -> list[str]:
        cmds = ["look"]
        if self._state["has_mug"]:
            if self._state["drawer_open"]:
                cmds.append("put mug in drawer")
        else:
            cmds.append("take mug")
        if self._state["drawer_open"]:
            cmds.append("close drawer")
        else:
            cmds.append("open drawer")
        return cmds

    def get_current_obs(self) -> dict:
        """Return obs dict for the current env state. Needed by brancher —
        after restore(), advisee needs to see the current obs to pick an action."""
        return self._build_obs_dict()

    def _build_obs_dict(self) -> dict:
        parts = ["kitchen scene."]
        if self._state["drawer_open"]:
            parts.append("Drawer is open.")
        else:
            parts.append("Drawer is closed.")
        if self._state["has_mug"]:
            parts.append("You are holding the mug.")
        elif self._state["mug_placed"]:
            parts.append("Mug is in the drawer.")
        else:
            parts.append("Mug is on the counter.")
        obs = ALFObs(
            text=" ".join(parts),
            task_description="put the mug in the drawer",
            admissible_commands=self._compute_admissible(),
            inventory_text=("You are carrying: mug." if self._state["has_mug"]
                             else "You are carrying: nothing."),
            step_count=self._step_count,
            task_type="mock_pick_and_place",
        )
        return obs.to_dict()
