"""
Minecraft prompt builders for NIC.

Text-mode (Option B) implementation: both advisee and companion consume the
structured *text* observation produced by MCEnv.get_current_obs() (scene /
held / inventory / vitals + milestones_reference). A future vision (Option A)
variant would add image_url content from the frames; the text schema here is
the fallback and the smoke-test path.

Contracts (kept identical to prompts/alfworld.py so advisee.py + VllmCompanionRaw
parse them unchanged):
  * advisee  -> emits exactly ONE mindcraft skill call, e.g. !collectBlocks("oak_log", 2)
  * companion -> emits JSON {"gate": "HELP"|"SILENCE", "advice": "<text>"}
"""

from __future__ import annotations

from typing import List, Optional

SYSTEM_PROMPT_ADVISEE = (
    "You are an expert Minecraft survival player using the mindcraft skill API. "
    "Given the task and current state, choose exactly ONE skill call to execute next. "
    "Respond with only the skill call (starting with '!'), nothing else."
)

SYSTEM_PROMPT_COMPANION = (
    "You are a Minecraft coach helping a player complete a task. At each step you "
    "decide whether to intervene. If the player is on track, stay silent. If the "
    "player appears stuck or is having difficulty, give a brief concrete hint. "
    'Respond with JSON: {"gate": "HELP" or "SILENCE", "advice": "<short next-step advice, or empty>"}.'
)


_SKILLS = [
    '!collectBlocks("<block>", <n>)   gather blocks/ores (oak_log, stone, coal_ore, iron_ore ...)',
    '!craftRecipe("<item>", <n>)      craft an item (needs ingredients; a crafting_table for 3x3 recipes)',
    '!smeltItem("<item>", <n>)        smelt in a nearby furnace (raw_iron -> iron_ingot; needs fuel like coal)',
    '!placeHere("<block>")            place a block from inventory (crafting_table, furnace)',
    '!searchForBlock("<block>", <r>)  locate the nearest block within range r',
    '!searchForEntity("<entity>", <r>) find and walk to the nearest mob within range r (e.g. sheep, cow)',
    '!useOn("<tool>", "<target>")     use a held tool on the nearest target (shears on sheep -> wool; bucket on cow -> milk)',
    '!equip("<item>")                 equip a tool (e.g. a pickaxe before mining)',
    '!digDown(<n>)                    dig straight down n blocks to reach stone/ores',
    '!goToCoordinates(<x>,<y>,<z>)    walk to a position',
    '!moveAway(<dist>)                move away from the current spot',
]

FEW_SHOT_ADVISEE: List = [
    {
        "user": (
            "Task: mine iron and craft an iron pickaxe\n"
            "Current state:\nHeld: nothing\nInventory: oak_log\n\n"
            'Reply in JSON: {"thoughts": "...", "action": "!skill(args)"}'
        ),
        "assistant": (
            '{"thoughts": "I have logs but no planks; craft planks first.", '
            '"action": "!craftRecipe(\\"oak_planks\\", 4)"}'
        ),
    },
    {
        "user": (
            "Task: mine iron and craft an iron pickaxe\n"
            "A coach suggests: Craft a wooden pickaxe before trying to mine stone.\n"
            "Current state:\nHeld: nothing\nInventory: oak_planks, stick, crafting_table\n\n"
            'Reply in JSON: {"thoughts": "...", "action": "!skill(args)"}'
        ),
        "assistant": (
            '{"thoughts": "I have planks, sticks and a table; craft the wooden pickaxe.", '
            '"action": "!craftRecipe(\\"wooden_pickaxe\\", 1)"}'
        ),
    },
]


def _state_block(obs: dict, show_subgoals: bool = True) -> str:
    """Render the MCEnv text observation into a compact state description.

    `show_subgoals`: include the milestone 'Subgoal path'. TRUE for the companion
    (it owns the reference plan); FALSE for the advisee/player, so the milestone
    plan stays companion-only (asymmetric information — NIC's core setup)."""
    if obs.get("text"):
        state = obs["text"]
    else:  # reconstruct from fields
        inv = ", ".join(obs.get("inventory", []) or []) or "empty"
        state = (f"Held: {obs.get('held_item') or 'nothing'}\n"
                 f"Inventory: {inv}\n"
                 f"Health: {obs.get('health')}  Food: {obs.get('food')}")
    ms = obs.get("milestones_reference") or []
    if ms and show_subgoals:
        state += "\nSubgoal path: " + " -> ".join(m.split(":")[0].strip() for m in ms)
    return state


def _history_block(history: Optional[list], window: int) -> str:
    if not history:
        return ""
    recent = history[-window:]
    lines = []
    for h in recent:
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            lines.append(f"  {h[0]} -> {h[1]}")
        else:
            lines.append(f"  {h}")
    return "Recent steps:\n" + "\n".join(lines) + "\n" if lines else ""


def build_advisee_prompt(
    obs: dict,
    advice: Optional[str] = None,
    history: Optional[list] = None,
    history_window: int = 8,
) -> str:
    """Prompt the advisee (frozen player) to emit one mindcraft skill call."""
    task = obs.get("task_description", "") or "survive and progress"
    parts = [
        f"Task: {task}",
        "",
        "Current state:",
        _state_block(obs, show_subgoals=False),
        "",
    ]
    hist = _history_block(history, history_window)
    if hist:
        parts += [hist]
    parts += [
        "Available skills:",
        *[f"  {s}" for s in _SKILLS],
        "",
    ]
    if advice:
        parts += [f">>> COACH HINT (follow this): {advice}",
                  "The coach can see more than you. This turn, PRIORITIZE the hint: "
                  "choose the single skill call that best carries it out (if it names a "
                  "resource, gather/craft toward it; if it implies a prerequisite, do "
                  "that first). Only ignore the hint if it is impossible in the current "
                  "state.", ""]
    parts += [
        "Think briefly about the best next step, then choose exactly ONE skill call. "
        'Reply in JSON: {"thoughts": "<one short sentence>", "action": "!skill(args)"}',
    ]
    return "\n".join(parts)


def build_companion_prompt(
    obs: dict,
    history: Optional[list] = None,
    history_window: int = 10,
    focus: Optional[str] = None,
) -> str:
    """Prompt the companion (coach) to decide gate + advice for the player.

    `focus`: optional subgoal to anchor the advice to (used for the K
    milestone-anchored fresh-advice branches; ignored on the main path).
    """
    task = obs.get("task_description", "") or "survive and progress"
    parts = [
        f"Task the player is working on: {task}",
        "",
        "Player's current state:",
        _state_block(obs),
        "",
    ]
    hist = _history_block(history, history_window)
    if hist:
        parts += [hist]
    if focus:
        parts += [f"Anchor your advice to helping the player reach this subgoal: {focus}", ""]
    frames = obs.get("frames_b64") or []
    if frames:
        parts.insert(2, "The image(s) show the player's current view. Ground your "
                        "advice in what is actually visible; do not invent objects.")
    parts += [
        "The player acts through high-level skills (collect, craft, smelt, place, "
        "search, equip, dig), not manual mouse/keyboard controls. Advise WHAT to do "
        "next (which resource to gather or item to craft/smelt), never how to click "
        "or which key to press.",
        "Decide whether to intervene now. If the player is progressing fine, "
        "gate=SILENCE with empty advice. If the player appears stuck or is having "
        "difficulty (repeated failed/ineffective actions, looping, or a missing "
        "prerequisite), gate=HELP with a brief concrete next-step hint (one "
        "sentence, actionable).",
        'Respond with JSON only: {"gate": "HELP"|"SILENCE", "advice": "..."}',
    ]
    text = "\n".join(parts)

    # Multimodal (Option A / VL companion): return OpenAI content parts with the
    # player's view image(s) + the text. If no frames, return plain text (Option B).
    if frames:
        content = [{"type": "text", "text": text}]
        for b in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b}"},
            })
        return content
    return text


class McPrompts:
    system_prompt_advisee = SYSTEM_PROMPT_ADVISEE
    system_prompt_companion = SYSTEM_PROMPT_COMPANION
    few_shot_advisee = FEW_SHOT_ADVISEE

    def build_advisee_prompt(self, *args, **kwargs):
        return build_advisee_prompt(*args, **kwargs)

    def build_companion_prompt(self, *args, **kwargs):
        return build_companion_prompt(*args, **kwargs)
