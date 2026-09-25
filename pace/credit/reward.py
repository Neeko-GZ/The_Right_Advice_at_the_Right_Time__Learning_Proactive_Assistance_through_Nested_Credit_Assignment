"""
reward.py — v2 causal-value reward for SPARK-Companion.

Per method.md § 3 the reward is:

    r_t = Σ_m 1[milestone m first-fires at t]         (task progress, ground truth)
        + R_success or R_failure at terminal step     (task outcome)
        − c · 1[HELP]                                 (per-intervention cost)
        − λ · |w| · 1[HELP]                           (advice-length penalty when HELP)

Note that unlike v1 there is NO advice-quality 2×2 term and NO SPARK Δ term.
Advice quality is credited by the causal estimator (b_help − V_silence) at
policy-optimization time, not baked into the environment reward. See
method.md § 4.2 for the split between reward and credit.

The 4 components below are additive; each returns its scalar contribution
and a dict of diagnostic keys under `r/*`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Protocol


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    # ── Milestone reward (ground-truth subtask progress) ──────────────
    milestone_coef: float = 1.0

    # ── Terminal reward (task outcome) ────────────────────────────────
    R_success: float = 1.0
    R_failure: float = 0.0

    # ── Per-intervention cost c ───────────────────────────────────────
    # c is a true per-intervention cost in task-reward units (a fraction
    # of a milestone). An intervention must buy ≥ c of expected progress
    # to be worth it. Also serves as the operating-point threshold in
    # the decide rule (HELP iff b_help − V_silence > c); swept in E1.
    intervention_cost: float = 0.1

    # ── Advice-length penalty λ ───────────────────────────────────────
    # Applied only when decision = HELP. Encourages concise advice
    # without penalizing silence.
    length_lambda: float = 0.005
    length_normalizer: str = "words"     # "words" or "chars"

    def asdict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# MilestoneDetector protocol
# ---------------------------------------------------------------------------
#
# reward_milestone used to hardcode an inventory check (Minecraft-style).
# For v2 we support multiple env backends:
#
#   InventoryDetector  → MC (inventory dict)               O(1) exact
#   ALFInfoDetector    → ALFWorld (info["won"] + subgoals) O(1) exact
#   VLMDetector        → Gaming-500 offline (VLM judge)    ~1s slow
#
# All must implement one method: `check(obs, milestone_spec) -> bool`,
# meaning "does the current observation satisfy this milestone's
# criterion?" This is called ONCE per milestone per step; caller
# (reward_milestone) handles the fire-once bookkeeping via
# `seen_milestones`.
# ---------------------------------------------------------------------------

class MilestoneDetector(Protocol):
    def check(self, obs: Any, milestone) -> bool:
        """Return True iff `obs` satisfies `milestone`'s criterion.

        `obs` can be a state dict (MC), an info dict (ALFWorld), or a
        (frame, task_desc) tuple (VLM). Detector implementations know
        which shape to expect.
        `milestone` is a task_pool.Milestone (has `.label`, `.target_items`,
        `.coef`); detectors may look at whatever fields they need.
        """
        ...


class InventoryDetector:
    """Minecraft-style: check whether any `target_items` appears with count>0
    in `state["inventory_top"]` or `state["held_item"]`."""

    def check(self, obs, milestone) -> bool:
        state = obs
        if not isinstance(state, dict):
            return False
        target_set = set(milestone.target_items)
        held = state.get("held_item") or {}
        if isinstance(held, dict) and held.get("name") in target_set:
            if int(held.get("count", 0) or 0) > 0:
                return True
        inv = state.get("inventory_top") or []
        if isinstance(inv, list):
            for it in inv:
                if isinstance(it, dict) and it.get("name") in target_set:
                    if int(it.get("count", 0) or 0) > 0:
                        return True
        return False


class ALFInfoDetector:
    """ALFWorld-style: check the env's `info["won"]` for task completion,
    plus optional per-milestone predicate on `info["facts"]` or similar.

    W28 EXP-005 confirmed `info["won"]` is available; per-milestone
    intermediate detection is optional (final `task_complete` milestone
    only for now — finer granularity comes in W29-W30)."""

    def check(self, obs, milestone) -> bool:
        info = obs
        if not isinstance(info, dict):
            return False
        # For the coarse "task_complete" milestone, use info["won"].
        # ALFWorld returns won batched: [True]/[False] — unwrap before bool().
        if milestone.label == "task_complete":
            v = info.get("won", False)
            if isinstance(v, (list, tuple)):
                return bool(v[0]) if v else False
            return bool(v)
        return False


class ALFTrajPlanDetector:
    """ALFWorld traj_data.json plan.high_pddl → auto-generated milestones.

    For each ALFWorld task instance, its `traj_data.json` contains
    `plan.high_pddl`: an ordered list of high-level PDDL actions
    (GotoLocation, PickupObject, PutObject, CleanObject, ...) that
    solve the task. We keep only "semantic" actions as milestones
    (Pickup / Put / Clean / Heat / Cool / Slice / Toggle) — GotoLocation
    and End are filtered.

    Usage from ALFWorldEnv:
        detector = ALFTrajPlanDetector()
        # on reset:
        milestones = detector.load_traj(info["extra.gamefile"])
        # on step:
        detector.observe_action(action_str)
        # in reward loop:
        detector.check(obs_or_info, milestone) → bool

    Milestone criterion encoding:
      ("traj_plan_step", int_semantic_idx)  → match by agent action history
      ("info", "won")                         → info["won"] flag
    """

    SEMANTIC_ACTIONS = {
        "PickupObject", "PutObject", "CleanObject",
        "HeatObject", "CoolObject", "SliceObject", "ToggleObject",
    }

    # High-level action → milestone verb + reward coef.
    # UNIFORM milestone reward: every subgoal completion grants the same
    # fixed coefficient (MILESTONE_COEF), so we do not hand-weight subgoal
    # types. Terminal task success is a separate, larger signal (the raw
    # env reward). This keeps the reward un-tuned and defensible.
    MILESTONE_COEF = 0.3
    _VERB_COEF = {
        "PickupObject":  ("picked_up",  MILESTONE_COEF),
        "PutObject":     ("placed",     MILESTONE_COEF),
        "CleanObject":   ("cleaned",    MILESTONE_COEF),
        "HeatObject":    ("heated",     MILESTONE_COEF),
        "CoolObject":    ("cooled",     MILESTONE_COEF),
        "SliceObject":   ("sliced",     MILESTONE_COEF),
        "ToggleObject":  ("toggled",    MILESTONE_COEF),
    }

    # Agent action-string tokens per high-level action.
    # ALFWorld emits several synonyms across versions; keep all seen forms.
    #   "move X to Y" is what the handcoded expert emits for PutObject
    #   (verified in W28 EXP-024). "put X in/on Y" is the free-text form.
    _ACTION_TOKENS = {
        "PickupObject": ("take", "pick up", "grab"),
        "PutObject":    ("put", "place", "move", "insert"),
        "CleanObject":  ("clean", "wash"),
        "HeatObject":   ("heat", "microwave"),
        "CoolObject":   ("cool", "chill"),
        "SliceObject":  ("slice", "cut"),
        "ToggleObject": ("turn on", "toggle", "switch on"),
    }

    def __init__(self):
        self._plan_steps: list[dict] = []
        self._action_history: list[str] = []

    # ── milestone-list construction ────────────────────────────────

    def load_traj(self, gamefile_path: str):
        """Read traj_data.json next to the gamefile and produce a list of
        Milestone objects. Also resets internal action history."""
        from pathlib import Path
        import json
        # Deferred import to avoid a circular dep at module load
        from rl_causal.task_pool import Milestone

        if not gamefile_path:
            self._plan_steps = []
            self._action_history = []
            return []

        trial_dir = Path(gamefile_path).parent
        traj_path = trial_dir / "traj_data.json"
        if not traj_path.exists():
            self._plan_steps = []
            self._action_history = []
            return []
        with traj_path.open("r") as f:
            traj = json.load(f)

        plan = traj.get("plan", {}).get("high_pddl", []) or []
        return self._build_milestones_from_plan(plan)

    def _build_milestones_from_plan(self, plan_high_pddl):
        """Split out for testability — accepts already-loaded plan list."""
        from rl_causal.task_pool import Milestone

        milestones: list = []
        self._plan_steps = []
        self._action_history = []
        semantic_idx = 0

        for raw_step in plan_high_pddl:
            planner_action = raw_step.get("planner_action") \
                if isinstance(raw_step, dict) else None
            if not isinstance(planner_action, dict):
                continue
            action = planner_action.get("action", "")
            if action not in self.SEMANTIC_ACTIONS:
                continue

            obj_id = str(planner_action.get("objectId") or "")
            obj_type = obj_id.split("|", 1)[0].lower()
            recep_id = str(planner_action.get("receptacleObjectId") or "")
            recep_type = recep_id.split("|", 1)[0].lower()

            verb, coef = self._VERB_COEF[action]
            label = (f"{verb}_{obj_type}_{semantic_idx}"
                     if obj_type else f"{verb}_{semantic_idx}")

            # Count prior plan_steps with same signature — needed to
            # distinguish e.g. "1st pen pickup" from "2nd pen pickup":
            # only fire this step's milestone once agent has performed
            # ≥ (occurrence_before + 1) matching actions in history.
            occurrence_before = sum(
                1 for p in self._plan_steps
                if p["action"] == action
                and p["obj_type"] == obj_type
                and p["recep_type"] == recep_type
            )
            self._plan_steps.append({
                "action": action,
                "obj_type": obj_type,
                "recep_type": recep_type,
                "semantic_index": semantic_idx,
                "occurrence_before": occurrence_before,
            })
            milestones.append(Milestone(
                label=label,
                target_items=(),
                coef=coef,
                criterion=("traj_plan_step", semantic_idx),
            ))
            semantic_idx += 1

        # Terminal task_complete driven by info["won"]
        milestones.append(Milestone(
            label="task_complete",
            target_items=(),
            coef=1.0,
            criterion=("info", "won"),
        ))
        return milestones

    # ── rollout-time hooks ─────────────────────────────────────────

    def observe_action(self, action_str: str) -> None:
        if action_str:
            self._action_history.append(action_str.lower())

    def reset_history(self) -> None:
        self._action_history = []

    # ── MilestoneDetector.check ────────────────────────────────────

    @staticmethod
    def _get_won(info) -> bool:
        """ALFWorld returns won as `[True]` / `[False]` (batched);
        unwrap before boolean coercion — `bool([False])` == True is
        a classic trap and made task_complete fire every step."""
        if not isinstance(info, dict):
            return False
        v = info.get("won", False)
        if isinstance(v, (list, tuple)):
            return bool(v[0]) if v else False
        return bool(v)

    def check(self, obs, milestone) -> bool:
        criterion = getattr(milestone, "criterion", None)
        if not isinstance(criterion, tuple) or not criterion:
            # Fallback: coarse task_complete label via info["won"]
            if getattr(milestone, "label", "") == "task_complete":
                return self._get_won(obs)
            return False

        kind = criterion[0]

        if kind == "info" and len(criterion) >= 2 and criterion[1] == "won":
            return self._get_won(obs)

        if kind == "traj_plan_step" and len(criterion) >= 2:
            sem_idx = int(criterion[1])
            if sem_idx < 0 or sem_idx >= len(self._plan_steps):
                return False
            return self._match_action(self._plan_steps[sem_idx])

        return False

    # ── action-string matching ─────────────────────────────────────

    def _match_action(self, plan_step: dict) -> bool:
        """Fire this step's milestone iff agent history contains at least
        (occurrence_before + 1) actions matching the step's signature.
        This distinguishes "1st pen pickup" from "2nd pen pickup"."""
        action = plan_step["action"]
        obj_type = plan_step["obj_type"]
        recep_type = plan_step["recep_type"]
        needed = int(plan_step.get("occurrence_before", 0)) + 1
        tokens = self._ACTION_TOKENS.get(action, ())

        matches = 0
        for a in self._action_history:
            if not any(tok in a for tok in tokens):
                continue
            if obj_type and obj_type not in a:
                continue
            if action == "PutObject" and recep_type and recep_type not in a:
                continue
            matches += 1
            if matches >= needed:
                return True
        return False


# ---------------------------------------------------------------------------
# Legacy convenience wrapper for MC inventory check
# ---------------------------------------------------------------------------

def _has_any_item(state: Optional[dict], target_names) -> bool:
    """Legacy convenience wrapper. Prefer InventoryDetector.check()."""
    if not isinstance(state, dict):
        return False
    target_set = set(target_names)
    held = state.get("held_item") or {}
    if isinstance(held, dict) and held.get("name") in target_set:
        if int(held.get("count", 0) or 0) > 0:
            return True
    inv = state.get("inventory_top") or []
    if isinstance(inv, list):
        for it in inv:
            if isinstance(it, dict) and it.get("name") in target_set:
                if int(it.get("count", 0) or 0) > 0:
                    return True
    return False


# ---------------------------------------------------------------------------
# component 1: milestone (ground-truth subtask progress)
# ---------------------------------------------------------------------------

def reward_milestone(
    state_now: Optional[Any],
    task_spec,
    seen_milestones: set,
    cfg: "RewardConfig",
    detector: Optional[MilestoneDetector] = None,
) -> tuple[float, dict]:
    """Fire each milestone at most once per episode.

    Caller must persist `seen_milestones` across steps within an episode,
    and clear it at episode start. `detector` defaults to InventoryDetector
    (MC-style backward compat)."""
    fired_this_step: list[str] = []
    weighted_step = 0.0
    final_completed_step = False

    if task_spec is None or state_now is None:
        return 0.0, {
            "r/ms_fired": [],
            "r/ms_count_total": len(seen_milestones),
            "r/ms_weighted": 0.0,
            "r/ms_completed_final": False,
        }

    detector = detector or InventoryDetector()

    final_label = task_spec.final_milestone().label
    for ms in task_spec.milestones:
        if ms.label in seen_milestones:
            continue
        if detector.check(state_now, ms):
            fired_this_step.append(ms.label)
            seen_milestones.add(ms.label)
            weighted_step += float(ms.coef) * cfg.milestone_coef
            if ms.label == final_label:
                final_completed_step = True

    return weighted_step, {
        "r/ms_fired": fired_this_step,
        "r/ms_count_total": len(seen_milestones),
        "r/ms_weighted": float(weighted_step),
        "r/ms_completed_final": final_completed_step,
    }


# ---------------------------------------------------------------------------
# component 2: terminal reward
# ---------------------------------------------------------------------------

def reward_terminal(done: bool, success: bool, cfg: "RewardConfig") -> tuple[float, dict]:
    if not done:
        return 0.0, {
            "r/terminal_fired": False,
            "r/terminal_success": False,
            "r/terminal_weighted": 0.0,
        }
    value = cfg.R_success if success else cfg.R_failure
    return float(value), {
        "r/terminal_fired": True,
        "r/terminal_success": bool(success),
        "r/terminal_weighted": float(value),
    }


# ---------------------------------------------------------------------------
# component 3: per-intervention cost
# ---------------------------------------------------------------------------

def reward_intervention_cost(decision: str, cfg: "RewardConfig") -> tuple[float, dict]:
    if decision == "HELP":
        weighted = -float(cfg.intervention_cost)
    else:
        weighted = 0.0
    return weighted, {
        "r/cost_applied": decision == "HELP",
        "r/cost_weighted": weighted,
    }


# ---------------------------------------------------------------------------
# component 4: advice-length penalty
# ---------------------------------------------------------------------------

def reward_length_penalty(advice_text: str, decision: str, cfg: "RewardConfig") -> tuple[float, dict]:
    if decision != "HELP" or not advice_text:
        return 0.0, {"r/len_units": 0, "r/len_weighted": 0.0}
    if cfg.length_normalizer == "words":
        units = len(advice_text.split())
    elif cfg.length_normalizer == "chars":
        units = len(advice_text)
    else:
        raise ValueError(
            f"unknown length_normalizer {cfg.length_normalizer!r}; use 'words' or 'chars'"
        )
    weighted = -float(cfg.length_lambda) * float(units)
    return weighted, {
        "r/len_units": int(units),
        "r/len_weighted": weighted,
    }


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------

def assemble_reward(
    decision: str,
    advice_text: str,
    state_now: Optional[Any],
    task_spec,
    seen_milestones: Optional[set],
    done: bool = False,
    success: bool = False,
    cfg: Optional["RewardConfig"] = None,
    detector: Optional[MilestoneDetector] = None,
) -> tuple[float, dict]:
    """Sum the 4 additive components. `detector` — pluggable milestone check;
    defaults to InventoryDetector (MC-style)."""
    cfg = cfg or RewardConfig()
    if seen_milestones is None:
        seen_milestones = set()

    r_ms, log_ms = reward_milestone(state_now, task_spec, seen_milestones, cfg, detector=detector)
    r_term, log_term = reward_terminal(done, success, cfg)
    r_cost, log_cost = reward_intervention_cost(decision, cfg)
    r_len, log_len = reward_length_penalty(advice_text, decision, cfg)

    total = float(r_ms + r_term + r_cost + r_len)
    log = {**log_ms, **log_term, **log_cost, **log_len, "r/total": total}
    return total, log
