"""
ALFWorld-specific prompt builders.

Two prompt templates:
  - Advisee side: given (task, state, admissible, history, optional advice),
    produce a single action from admissible_commands
  - Companion side: given (task, state, admissible, history), produce
    step-by-step natural-language advice

Both templates share:
  - "Task: <turk_annotations.task_desc>"
  - "Current observation: <env text>"
  - "Inventory: ..."
  - "Admissible actions: ..." (ALFWorld-native, discrete)
  - "History: [<step> Action: <a>, Obs: <o>]"

Advisee-only: JSON output request + few-shot demonstrating {"thoughts", "action"}
Companion-only: numbered plan output request

Kept as top-level module functions (not a class) — cheaper to import,
avoids OOP overhead. `AlfworldPrompts` class at bottom bundles them for
users who prefer the PromptBuilder Protocol interface.
"""

from __future__ import annotations

import os
from typing import Optional

from rl_causal.prompts.base import summarize_obs

# v16 switches (default = original 56%-baseline behaviour; opt in via env):
#   NIC_PROMPT=new         -> fuller "concise ordered plan" companion prompt
#   NIC_FOCUS_MODE=horizon -> CF branches anchor to the next k milestones
#                             (horizon diversity) instead of a single one by index
_NIC_NEW_PROMPT = os.environ.get("NIC_PROMPT", "").lower() in ("new", "v2", "v16")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ADVISEE = "You are an expert in the ALFworld text Environment."

_COMPANION_CONTENT_V1 = (
    "  (2) CONTENT — if HELP, WHAT to say. Give staged tactical advice about "
    "the immediate next 1-2 actions only; do not dump the whole walkthrough.\n\n")
_COMPANION_CONTENT_V2 = (
    "  (2) CONTENT — if HELP, WHAT to say. Give concise, staged guidance for "
    "the player's current unresolved subgoal. Mention the key next operations "
    "and their ordering or dependencies when useful, rather than only the next "
    "single action. Advise what operations to perform and in what order (e.g., "
    "pick up, then cool/heat/clean, then place). Do not infer or invent an "
    "object's specific location, direction, or current container unless it is "
    "explicitly supported by the task, history, current observation, or "
    "reference plan. Keep the advice brief and do not provide the full walkthrough.\n\n")

SYSTEM_PROMPT_COMPANION = (
    "You are a companion assistant helping a player in the ALFworld text "
    "environment. At each turn you make TWO decisions:\n"
    "  (1) GATE — whether to speak ('HELP') or stay quiet ('SILENCE') this turn.\n"
    "      Choose SILENCE when the player is on-track and doesn't need "
    "interruption. Choose HELP only when they're stuck, off-track, or "
    "confused.\n"
    + (_COMPANION_CONTENT_V2 if _NIC_NEW_PROMPT else _COMPANION_CONTENT_V1)
    + "Always output STRICT JSON: "
    "{\"gate\": \"HELP\"|\"SILENCE\", \"advice\": \"...\"}. "
    "When SILENCE, set advice to \"\"."
)


# ---------------------------------------------------------------------------
# Few-shot for advisee (JSON output format)
# ---------------------------------------------------------------------------

FEW_SHOT_ADVISEE = [
    {
        "user": (
            "Task: put the apple on the table\n"
            "History: None\n"
            "Current observation: You are in a kitchen. On the counter you see an apple 1.\n"
            "Inventory: (nothing)\n"
            "Admissible actions:\n"
            "  - look\n"
            "  - go to counter 1\n"
            "  - go to table 1\n"
            "  - take apple 1 from counter 1\n\n"
            "Reply in JSON: {\"thoughts\": \"...\", \"action\": \"one admissible action\"}"
        ),
        "assistant": (
            '{"thoughts": "I see the apple on the counter. I should take it first.", '
            '"action": "take apple 1 from counter 1"}'
        ),
    },
    {
        "user": (
            "Task: heat some mug and put it in the coffeemachine\n"
            "Companion advice: The mug needs to be heated in the microwave before placing.\n"
            "History:\n"
            "  [0] Action: go to countertop 1, Obs: You see a mug 1.\n"
            "  [1] Action: take mug 1 from countertop 1, Obs: You take mug 1.\n"
            "Current observation: You are at microwave 1. The microwave 1 is closed.\n"
            "Inventory: mug 1\n"
            "Admissible actions:\n"
            "  - look\n"
            "  - open microwave 1\n"
            "  - go to coffeemachine 1\n\n"
            "Reply in JSON: {\"thoughts\": \"...\", \"action\": \"one admissible action\"}"
        ),
        "assistant": (
            '{"thoughts": "Advice says heat in microwave first. Open the microwave.", '
            '"action": "open microwave 1"}'
        ),
    },
]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_advisee_prompt(
    obs: dict,
    advice: Optional[str] = None,
    history: Optional[list] = None,
    history_window: int = 8,
) -> str:
    """Build the user turn for advisee inference.

    Reads standard fields from `obs`:
      - task_description : str
      - text             : str  (current env obs)
      - inventory_text   : str
      - admissible_commands : list[str]
      - step_count       : int  (for history step labels)

    `advice` is None for SILENCE branches, str for HELP branches.
    Only structural difference: presence of "Companion advice: ..." line.
    """
    task = obs.get("task_description", "") or ""
    text_obs = obs.get("text", "") or ""
    inventory = obs.get("inventory_text", "") or ""
    admissible = obs.get("admissible_commands", []) or []
    step_count = obs.get("step_count", 0)

    action_list = ("\n".join("  - " + a for a in admissible)
                   if admissible else "  (none)")

    hist_str = "None"
    if history:
        recent = history[-history_window:]
        lines = []
        for i, (act, ob) in enumerate(recent):
            true_step = step_count - len(recent) + i
            lines.append("  [" + str(true_step) + "] Action: " + act
                         + ", Obs: " + summarize_obs(ob))
        hist_str = "\n" + "\n".join(lines)

    has_advice = advice is not None and str(advice).strip()
    advice_block = ""
    follow_line = ""
    if has_advice:
        advice_block = (">>> A companion is helping you. HINT: "
                        + str(advice).strip() + "\n")
        follow_line = (
            "A companion hint is given above. Treat it as a trusted guide: choose the "
            "admissible action that best carries out the hint (if the hint names an "
            "object you must reach first, take the step that moves toward it). "
        )

    return (
        "Task: " + str(task) + "\n"
        + advice_block
        + "History: " + hist_str + "\n"
        "Current observation: " + str(text_obs) + "\n"
        "Inventory: " + (str(inventory) if inventory else "(nothing)") + "\n"
        "Admissible actions:\n"
        + action_list
        + "\n\n" + follow_line
        + "Complete the task in as few steps as possible. Break it into "
        + "subtasks and finish them step by step. Do not repeat an action "
        + "continuously. Do not invent objects that are not listed above.\n\n"
        + "Reply in JSON: {\"thoughts\": \"describe what you see and reason "
        + "about the next step\", \"action\": \"one admissible action\"}"
    )


def build_companion_prompt(
    obs: dict,
    history: Optional[list] = None,
    history_window: int = 10,
    focus: Optional[str] = None,
) -> str:
    """Build the user turn for companion inference / SFT training.

    Reads same obs schema as advisee (task_description, text, inventory_text,
    admissible_commands). Also formats history the same way. Output request
    differs: instead of a single JSON action, asks for a numbered plan.

    `focus` (optional): a subgoal/milestone string. When set, the companion is
    directed to focus THIS advice on that subgoal. Used to sample structured,
    milestone-anchored counterfactual advice branches (diverse yet coherent,
    env-agnostic — works in ALFWorld and Minecraft alike).
    """
    task = obs.get("task_description", "") or ""
    text_obs = obs.get("text", "") or ""
    inventory = obs.get("inventory_text", "") or ""
    admissible = obs.get("admissible_commands", []) or []
    # Asymmetric info: reference milestones (advisee does NOT see this).
    # From ALFTrajPlanDetector via env, human-readable form.
    milestones = obs.get("milestones_reference", []) or []

    action_list = ("\n".join("  - " + a for a in admissible)
                   if admissible else "  (none)")

    milestone_block = ""
    if milestones:
        ms_str = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(milestones))
        milestone_block = (
            f"Reference plan (high-level milestones for this task, "
            f"only YOU see this):\n{ms_str}\n"
        )

    hist_str = "None"
    if history:
        recent = history[-history_window:]
        lines = []
        for i, (act, ob) in enumerate(recent):
            lines.append(f"  [{i}] Action: {act}, Obs: {summarize_obs(ob)}")
        hist_str = "\n" + "\n".join(lines)

    focus_block = ""
    if focus:
        if isinstance(focus, (list, tuple)):
            # horizon mode: a range of upcoming subgoals -> ordered plan
            fs = "; ".join(str(f) for f in focus if str(f).strip())
            focus_block = (
                f"Focus THIS advice on the next subgoals, in order: {fs}\n"
                f"Give a concise ordered plan that carries the player through them.\n"
            )
        else:
            focus_block = (
                f"Focus THIS advice specifically on the subgoal: {focus}\n"
                f"Give concrete next-step guidance that moves the player toward it.\n"
            )

    _tail_v1 = (
        "Advice must be 1-3 sentences of staged, immediate next-step "
        "guidance based on the reference plan above — NOT the whole "
        "walkthrough. Give HELP only when player seems stuck or off-plan.")
    _tail_v2 = (
        "If HELP is chosen, give 1-3 concise sentences of staged, short-horizon "
        "guidance for the current subgoal. Include the key next operations and "
        "their ordering when useful. Do not guess where an object is located or "
        "which direction to search; only state locations that are explicitly "
        "supported by the provided context. Do not provide the full task walkthrough.")

    return (
        f"Task: {task}\n"
        f"{milestone_block}"
        f"History of player's actions: {hist_str}\n"
        f"Current observation: {text_obs}\n"
        f"Inventory: {inventory or '(nothing)'}\n"
        f"Admissible actions:\n{action_list}\n\n"
        f"{focus_block}"
        "Decide gate + (if HELP) advice. Reply with JSON: "
        "{\"gate\": \"HELP\"|\"SILENCE\", \"advice\": \"...\"}. "
        + (_tail_v2 if _NIC_NEW_PROMPT else _tail_v1)
    )


# ---------------------------------------------------------------------------
# Bundle as class (satisfies PromptBuilder Protocol)
# ---------------------------------------------------------------------------

class AlfworldPrompts:
    """Bundles ALFWorld prompt builders as a PromptBuilder-compatible class.
    Useful when passing a single object around. Otherwise import functions
    directly.
    """
    system_prompt_advisee = SYSTEM_PROMPT_ADVISEE
    system_prompt_companion = SYSTEM_PROMPT_COMPANION
    few_shot_advisee = FEW_SHOT_ADVISEE

    def build_advisee_prompt(self, obs, advice=None, history=None, history_window=8):
        return build_advisee_prompt(obs, advice, history, history_window)

    def build_companion_prompt(self, obs, history=None, history_window=10):
        return build_companion_prompt(obs, history, history_window)
