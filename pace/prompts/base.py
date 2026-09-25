"""
Base Protocol for env-specific prompt builders.

Not strictly enforced at runtime — modules just need to expose the
required names (system_prompt_advisee, few_shot_advisee, build_advisee_prompt,
build_companion_prompt). This Protocol is for documentation + optional type
checking.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class PromptBuilder(Protocol):
    """Env-specific prompt building interface.

    Attributes:
        system_prompt_advisee   : System prompt for advisee inference
        system_prompt_companion : System prompt for companion inference / SFT
        few_shot_advisee        : List of {"user": str, "assistant": str}
                                  demonstrating expected output format

    Methods:
        build_advisee_prompt : Build the user turn given obs + optional advice
        build_companion_prompt : Build the user turn for companion given obs
    """
    system_prompt_advisee: str
    system_prompt_companion: str
    few_shot_advisee: list

    def build_advisee_prompt(
        self,
        obs: dict,
        advice: Optional[str] = None,
        history: Optional[list] = None,
        history_window: int = 8,
    ) -> str: ...

    def build_companion_prompt(
        self,
        obs: dict,
        history: Optional[list] = None,
        history_window: int = 10,
    ) -> str: ...


def summarize_obs(text: str, max_chars: int = 120) -> str:
    """One-line brief of an observation for history logging. Shared helper
    since most prompt builders will need this."""
    if not text:
        return "(empty)"
    s = " ".join(str(text).split())
    return s if len(s) <= max_chars else s[:max_chars - 3] + "..."
