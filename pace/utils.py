"""
utils.py — small shared helpers used across rl_causal modules.

Kept intentionally minimal — this is not a dumping ground. Only things
that are used in ≥ 2 places (avoiding drift-prone duplication) belong here.
"""

from __future__ import annotations


def unwrap_won(info: dict) -> bool:
    """ALFWorld's info['won'] can be bool or [bool] (batched). Unwrap safely.

    Also used by pc1_responsiveness.py and brancher.py — deduplicating here
    keeps semantics identical across both.
    """
    if not isinstance(info, dict):
        return False
    w = info.get("won", False)
    if isinstance(w, (list, tuple)) and w:
        return bool(w[0])
    return bool(w)
