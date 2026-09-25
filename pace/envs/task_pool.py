"""
task_pool.py — Task specifications + LLM-decomposed subtask milestones.

Each task spec defines:
  - task_text:       Short natural-language description, fed to the policy
                     via PROMPT_TEMPLATE {task} and to SparkScorer as the
                     task-conditioning text.
  - goal_template:   The !goal(...) command sent to the bot at episode
                     reset to kick its autonomous play mode toward this task.
  - max_steps:       Per-task episode budget. Simple gathering tasks finish
                     in 5-10 steps; compound crafting tasks need 15-20.
  - milestones:      Ordered list of subtask checkpoints, each as
                     (label, target_items_tuple, coef). PPO gets a one-shot
                     reward when the bot first acquires any item in
                     target_items. Final milestone is the task goal itself
                     (highest coef). Earlier milestones are intermediate
                     resources / tools the LLM-decomposed planner expects
                     the bot to acquire en route.

Multi-step / compound tasks (pickaxe, furnace, bed) have 5-7 milestones
so PPO gets dense intermediate reward even when the final goal isn't
hit within the episode budget. This replaces the sparse "only-final-
reward" structure that made compound tasks essentially un-trainable.

Milestone curation methodology (write this in the paper):
> "Subtask decompositions are LLM-generated (Qwen3-VL prompted with
>  'list the items the survival agent must acquire in order to complete
>  task X') and curated for inventory-detectability. Each milestone
>  corresponds to one or more Minecraft item IDs that appear in the bot's
>  inventory upon completion."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


# All wood log variants — used as "wood" for any task that needs wood.
_ALL_LOGS = (
    "oak_log", "birch_log", "spruce_log", "jungle_log",
    "acacia_log", "dark_oak_log", "mangrove_log", "cherry_log",
)
_ALL_PLANKS = (
    "oak_planks", "birch_planks", "spruce_planks", "jungle_planks",
    "acacia_planks", "dark_oak_planks", "mangrove_planks", "cherry_planks",
)
_ALL_WOOL = (
    "white_wool", "orange_wool", "magenta_wool", "light_blue_wool",
    "yellow_wool", "lime_wool", "pink_wool", "gray_wool",
    "light_gray_wool", "cyan_wool", "purple_wool", "blue_wool",
    "brown_wool", "green_wool", "red_wool", "black_wool",
)
_ALL_BEDS = (
    "white_bed", "orange_bed", "magenta_bed", "light_blue_bed",
    "yellow_bed", "lime_bed", "pink_bed", "gray_bed",
    "light_gray_bed", "cyan_bed", "purple_bed", "blue_bed",
    "brown_bed", "green_bed", "red_bed", "black_bed",
)


@dataclass
class Milestone:
    """One subtask checkpoint. Triggered once per episode when its criterion
    (detector-specific) is met for the first time.

    Fields:
      label        : short name for logging.
      target_items : legacy — Minecraft-style item name list, consumed by
                     `InventoryDetector`. Empty tuple for non-MC envs.
      coef         : reward multiplier when the milestone fires.
      criterion    : detector-specific opaque predicate:
                       - MC:    unused (detector uses target_items)
                       - ALF info:      ("info", "won")
                       - ALF traj plan: ("traj_plan_step", int_idx)
                       - VLM:   ("vlm", "<natural language question>")
                     Each detector's `.check(obs, milestone)` interprets it.
    """
    label: str
    target_items: tuple[str, ...] = ()
    coef: float = 0.3
    criterion: Any = None


@dataclass
class TaskSpec:
    name: str                       # internal id, also used for logging
    task_text: str                  # fed to policy prompt + SparkScorer
    goal_template: str              # !goal command for bot autonomous mode
    max_steps: int                  # per-task episode length budget
    milestones: list[Milestone]     # subtask checkpoints (ordered)

    def final_milestone(self) -> Milestone:
        return self.milestones[-1]


# ---------------------------------------------------------------------------
# the 7-task pool
# ---------------------------------------------------------------------------
TASK_POOL: list[TaskSpec] = [
    # ─── Tier 1: atomic gathering tasks ─────────────────────────────────
    TaskSpec(
        name="collect_wood",
        task_text="collect wood from trees",
        goal_template='!goal("Play Minecraft and collect wood.")',
        max_steps=10,
        milestones=[
            Milestone("got wood", _ALL_LOGS, 0.3),
        ],
    ),
    TaskSpec(
        name="collect_dirt",
        task_text="dig up dirt blocks",
        goal_template='!goal("Play Minecraft and collect dirt.")',
        max_steps=10,
        milestones=[
            Milestone("got dirt", ("dirt", "grass_block", "coarse_dirt"), 0.1),
        ],
    ),
    TaskSpec(
        name="hunt_chicken",
        task_text="hunt chickens for food and feathers",
        goal_template='!goal("Hunt chickens for food.")',
        max_steps=10,
        milestones=[
            Milestone("got feather", ("feather",), 0.5),
            Milestone("got meat", ("chicken", "raw_chicken", "cooked_chicken"), 0.5),
        ],
    ),

    # ─── Tier 2: single-craft ───────────────────────────────────────────
    TaskSpec(
        name="craft_crafting_table",
        task_text="craft a crafting table",
        goal_template='!goal("Craft a crafting table.")',
        max_steps=15,
        milestones=[
            Milestone("got wood",   _ALL_LOGS,   0.2),
            Milestone("got planks", _ALL_PLANKS, 0.3),
            Milestone("got table",  ("crafting_table",), 1.0),
        ],
    ),

    # ─── Tier 3: multi-step compound ────────────────────────────────────
    TaskSpec(
        name="craft_wooden_pickaxe",
        task_text="craft a wooden pickaxe",
        goal_template='!goal("Craft a wooden pickaxe.")',
        max_steps=20,
        milestones=[
            Milestone("got wood",    _ALL_LOGS, 0.2),
            Milestone("got planks",  _ALL_PLANKS, 0.2),
            Milestone("got sticks",  ("stick",), 0.3),
            Milestone("got table",   ("crafting_table",), 0.3),
            Milestone("got pickaxe", ("wooden_pickaxe",), 2.0),
        ],
    ),
    TaskSpec(
        name="make_furnace",
        task_text="make a furnace by mining stone",
        goal_template='!goal("Make a furnace.")',
        max_steps=20,
        milestones=[
            Milestone("got wood",    _ALL_LOGS, 0.2),
            Milestone("got planks",  _ALL_PLANKS, 0.2),
            Milestone("got sticks",  ("stick",), 0.2),
            Milestone("got table",   ("crafting_table",), 0.3),
            Milestone("got pickaxe", ("wooden_pickaxe",), 0.5),
            Milestone("got stone",   ("cobblestone", "stone"), 0.5),
            Milestone("got furnace", ("furnace",), 2.5),
        ],
    ),
    TaskSpec(
        name="collect_stone",
        task_text="mine cobblestone (craft a pickaxe first)",
        goal_template='!goal("Mine cobblestone.")',
        max_steps=30,
        milestones=[
            Milestone("got wood",    _ALL_LOGS, 0.2),
            Milestone("got planks",  _ALL_PLANKS, 0.2),
            Milestone("got sticks",  ("stick",), 0.2),
            Milestone("got table",   ("crafting_table",), 0.3),
            Milestone("got pickaxe", ("wooden_pickaxe",), 0.5),
            Milestone("got stone",   ("cobblestone", "stone"), 2.0),
        ],
    ),
    TaskSpec(
        name="craft_stone_pickaxe",
        task_text="craft a stone pickaxe",
        goal_template='!goal("Craft a stone pickaxe.")',
        max_steps=30,
        milestones=[
            Milestone("got wood",      _ALL_LOGS, 0.2),
            Milestone("got planks",    _ALL_PLANKS, 0.2),
            Milestone("got sticks",    ("stick",), 0.2),
            Milestone("got table",     ("crafting_table",), 0.3),
            Milestone("got w.pickaxe", ("wooden_pickaxe",), 0.5),
            Milestone("got stone",     ("cobblestone", "stone"), 0.6),
            Milestone("got s.pickaxe", ("stone_pickaxe",), 2.5),
        ],
    ),

    # ─── Tier 4: cross-domain compound ──────────────────────────────────
    TaskSpec(
        name="craft_bed",
        task_text="craft a bed using wool and wood",
        goal_template='!goal("Craft a bed.")',
        max_steps=20,
        milestones=[
            Milestone("got wood",    _ALL_LOGS, 0.2),
            Milestone("got planks",  _ALL_PLANKS, 0.2),
            Milestone("got wool",    _ALL_WOOL, 1.0),
            Milestone("got table",   ("crafting_table",), 0.3),
            Milestone("got bed",     _ALL_BEDS, 3.0),
        ],
    ),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def get_task_by_name(name: str) -> TaskSpec:
    for t in TASK_POOL:
        if t.name == name:
            return t
    raise KeyError(f"task {name!r} not in TASK_POOL; available: {[t.name for t in TASK_POOL]}")


def sample_task(rng=None) -> TaskSpec:
    """Uniformly sample a task from the pool. Pass a numpy RandomState for
    deterministic episode-task assignment in evals."""
    import random
    if rng is not None:
        return rng.choice(TASK_POOL)
    return random.choice(TASK_POOL)


# ---------------------------------------------------------------------------
# Stratified sampling: simple tasks are the only ones the gpt-5-nano advisee
# bot can reliably complete in <=10 steps. To guarantee ms-fire signal each
# update (rather than relying on random.choice luck), we force at least one
# simple task into every batch.
# ---------------------------------------------------------------------------

SIMPLE_TASK_NAMES = ("collect_wood", "collect_dirt", "hunt_chicken")


def _simple_tasks() -> list[TaskSpec]:
    return [t for t in TASK_POOL if t.name in SIMPLE_TASK_NAMES]


def sample_tasks_stratified(n: int, rng=None) -> list[TaskSpec]:
    """Return n tasks with at least one drawn from SIMPLE_TASK_NAMES.

    Guarantees: if n >= 1, output[0] is a simple task; remaining
    n-1 slots are uniformly sampled from the full TASK_POOL.

    Rationale: during early PPO smoke / training, hard tasks
    (craft_*, make_furnace) produce zero milestone fires because the
    advisee bot cannot complete them in the budget; that gives a
    monotone-bad reward signal that inflates KL. One guaranteed simple
    task per batch keeps the milestone reward non-zero per update.
    """
    import random
    if n <= 0:
        return []
    simple_pool = _simple_tasks()
    if rng is not None:
        first = rng.choice(simple_pool)
        rest = [rng.choice(TASK_POOL) for _ in range(n - 1)]
    else:
        first = random.choice(simple_pool)
        rest = [random.choice(TASK_POOL) for _ in range(n - 1)]
    return [first] + rest


def all_task_names() -> list[str]:
    return [t.name for t in TASK_POOL]
