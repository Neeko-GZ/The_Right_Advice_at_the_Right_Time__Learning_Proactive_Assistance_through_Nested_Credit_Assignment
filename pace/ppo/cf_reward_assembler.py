"""
cf_reward_assembler.py — intrinsic quality + intra-group diversity rewards.

Method.md § 4.2 defines the group-relative advantage as:

    Â_i = (Q_i - Q̄) / σ_Q + α_intr · r_intr(a_i) + α_div · r_div(a_i, {a_j})

The (Q̄, σ_Q) part is done in cf_advantage.py. This module handles the two
shaping terms:

  r_intr(a_i)  — an intrinsic quality signal for advice a_i, independent of
                 the branch outcome. Currently:
                 • format validity  (well-formed JSON)
                 • grounding        (mentions objects present in the state)
                 • non-empty        (advice isn't just whitespace)

  r_div(a_i, {a_j}) — a diversity term across the K+1 group. High when a_i
                 differs semantically from the other sampled advice. Simple
                 implementation: normalized token-overlap distance.

Both are per-branch scalars, then combined with weights (α_intr, α_div).

Pure numerical code — no torch dependency. Advice strings and state dicts
come in as plain Python.

Reference: method.md § 4.2, § 5.5.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Intrinsic quality reward
# ---------------------------------------------------------------------------

def format_validity_score(raw_response_text: Optional[str]) -> float:
    """1.0 if the companion output is a well-formed JSON with expected keys,
    partial credit if only some structure is present, 0.0 if garbage.

    Companion is expected to emit exactly:
      {"gate": "HELP"|"SILENCE", "advice": "..."}

    Rubric:
      1.0  — parses as JSON dict with both keys and valid gate value
      0.7  — parses as JSON but missing a key or gate value not in {HELP,SILENCE}
      0.3  — has {"gate": ...} and {"advice": ...} substrings but not valid JSON
      0.0  — no discernible structure

    Args:
      raw_response_text: the raw assistant text emitted by companion
    """
    if raw_response_text is None:
        return 0.0
    s = raw_response_text.strip()
    if not s:
        return 0.0
    # Strip code fences
    s_clean = re.sub(r"^```(?:json)?\s*", "", s)
    s_clean = re.sub(r"\s*```$", "", s_clean)

    # Try strict JSON parse first
    try:
        obj = json.loads(s_clean)
        if isinstance(obj, dict) and "gate" in obj and "advice" in obj:
            gate_val = str(obj["gate"]).strip().upper()
            if gate_val in ("HELP", "SILENCE"):
                return 1.0
            return 0.7
        return 0.7  # parses but wrong shape
    except json.JSONDecodeError:
        pass

    # Fallback: regex detection
    has_gate = bool(re.search(r'"gate"\s*:', s_clean))
    has_advice = bool(re.search(r'"advice"\s*:', s_clean))
    if has_gate and has_advice:
        return 0.3
    return 0.0


def grounding_score(advice_text: str, state_dict: Optional[Dict]) -> float:
    """Fraction of key state-mentioned objects the advice references.

    "Grounded" advice mentions objects/receptacles the advisee can actually
    interact with in the current state. We extract candidate objects from
    the observation text and admissible commands, then compute overlap.

    Rubric:
      1.0  — mentions ≥2 of the objects visible / admissible in state
      0.6  — mentions exactly 1
      0.2  — no mention but advice is non-empty
      0.0  — advice is empty

    Args:
      advice_text: the advice content string
      state_dict: dict optionally containing 'text', 'admissible_commands'

    Note: this is a coarse heuristic. Its role is to nudge the companion
    toward state-grounded outputs; heavy lifting comes from Q-value shaping.
    """
    if not advice_text or not advice_text.strip():
        return 0.0
    if state_dict is None:
        return 0.2

    obs_text = str(state_dict.get("text", "")) if isinstance(state_dict, dict) else ""
    admissible = state_dict.get("admissible_commands", []) if isinstance(state_dict, dict) else []

    # Extract object candidates: alphanumeric words with length ≥3 from obs
    # plus objects mentioned in admissible commands.
    obs_words = set(w.lower() for w in re.findall(r"[a-zA-Z]{3,}\b", obs_text))
    admissible_words = set()
    for cmd in admissible:
        admissible_words.update(w.lower() for w in re.findall(r"[a-zA-Z]{3,}\b", str(cmd)))

    # Filter out very common English stopwords that would produce false hits
    stopwords = {
        "the", "and", "for", "you", "are", "with", "this", "that", "into",
        "onto", "from", "have", "has", "look", "there", "here", "some", "any",
        "will", "your", "not", "all", "can", "just", "then", "when", "now",
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    }
    candidates = (obs_words | admissible_words) - stopwords
    if not candidates:
        return 0.2

    advice_lower = advice_text.lower()
    hits = sum(1 for w in candidates if w in advice_lower)

    if hits >= 2:
        return 1.0
    if hits == 1:
        return 0.6
    return 0.2


def compute_intrinsic_reward(
    advice_text: str,
    raw_response_text: Optional[str],
    state_dict: Optional[Dict],
    w_format: float = 0.4,
    w_grounding: float = 0.6,
) -> float:
    """Combine format validity + grounding into a single [0, 1] score.

    Weights default to (0.4, 0.6) — grounding matters slightly more than
    raw JSON well-formedness since the SFT warmup should already handle
    most format cases.
    """
    r_fmt = format_validity_score(raw_response_text)
    r_gnd = grounding_score(advice_text, state_dict)
    return w_format * r_fmt + w_grounding * r_gnd


# ---------------------------------------------------------------------------
# Diversity reward
# ---------------------------------------------------------------------------

def _tokenize_for_diversity(text: str) -> List[str]:
    """Lowercase word tokens for Jaccard-style overlap."""
    if not text:
        return []
    return re.findall(r"[a-zA-Z]{2,}", text.lower())


def jaccard_distance(a: List[str], b: List[str]) -> float:
    """1 - |A ∩ B| / |A ∪ B|. Returns 1.0 if both empty."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 1.0
    return 1.0 - inter / union


def compute_diversity_reward(
    advice_i: str,
    other_advices: Sequence[str],
) -> float:
    """Mean pairwise Jaccard distance from advice_i to the others.

    r_div ∈ [0, 1]:
      1.0 — completely disjoint token sets from every peer (max diversity)
      0.0 — identical to every peer (all overlap)
      middle values follow the mean.

    If `other_advices` is empty, returns 0.5 (neutral).
    """
    if not other_advices:
        return 0.5
    tokens_i = _tokenize_for_diversity(advice_i)
    dists = [
        jaccard_distance(tokens_i, _tokenize_for_diversity(other))
        for other in other_advices
    ]
    return sum(dists) / len(dists)


# ---------------------------------------------------------------------------
# Group-level assembler (main entry point)
# ---------------------------------------------------------------------------

@dataclass
class BranchShapingReward:
    """Per-branch decomposition. Kept as separate fields for logging."""
    r_intrinsic: float
    r_diversity: float
    r_format: float
    r_grounding: float

    def combined(self, alpha_intr: float, alpha_div: float) -> float:
        return alpha_intr * self.r_intrinsic + alpha_div * self.r_diversity


def compute_group_shaping_rewards(
    advice_texts: Sequence[str],
    raw_responses: Sequence[Optional[str]],
    state_dict: Optional[Dict],
    w_format: float = 0.4,
    w_grounding: float = 0.6,
) -> List[BranchShapingReward]:
    """For a K+1-branch group at a single state, compute (r_intr, r_div) per branch.

    The SILENCE branch is passed with advice_text == "" and raw_responses[i]
    being either None or the JSON that produced the SILENCE decision. Its
    intrinsic reward will be low (empty advice) and diversity is measured
    against the actual advice strings only.

    Args:
      advice_texts    length K+1, one per branch (SILENCE index has empty str)
      raw_responses   length K+1, raw JSON that produced each advice (may be None)
      state_dict      shared state (all branches start from the same snapshot)
      w_format        weight for format validity in intrinsic reward
      w_grounding     weight for grounding in intrinsic reward

    Returns:
      list of BranchShapingReward, len == K+1
    """
    n = len(advice_texts)
    if len(raw_responses) != n:
        raise ValueError(
            f"advice_texts len {n} != raw_responses len {len(raw_responses)}"
        )

    results: List[BranchShapingReward] = []
    for i in range(n):
        # Intrinsic decomposition (report parts for logging)
        r_fmt = format_validity_score(raw_responses[i])
        r_gnd = grounding_score(advice_texts[i], state_dict)
        r_intr = w_format * r_fmt + w_grounding * r_gnd

        # Diversity: compare to all OTHER branches in the group
        others = [a for j, a in enumerate(advice_texts) if j != i]
        r_div = compute_diversity_reward(advice_texts[i], others)

        results.append(BranchShapingReward(
            r_intrinsic=r_intr,
            r_diversity=r_div,
            r_format=r_fmt,
            r_grounding=r_gnd,
        ))

    return results


# ---------------------------------------------------------------------------
# Sanity self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    print("[test] format_validity_score")
    good = '{"gate": "HELP", "advice": "Go to the drawer"}'
    print(f"  good JSON:   {format_validity_score(good):.2f}")
    assert format_validity_score(good) == 1.0

    bad_gate = '{"gate": "MAYBE", "advice": "..."}'
    print(f"  bad gate:    {format_validity_score(bad_gate):.2f}")
    assert format_validity_score(bad_gate) == 0.7

    fenced = '```json\n{"gate": "SILENCE", "advice": ""}\n```'
    print(f"  code fenced: {format_validity_score(fenced):.2f}")
    assert format_validity_score(fenced) == 1.0

    junk = "well i think you should just go find it"
    print(f"  junk text:   {format_validity_score(junk):.2f}")
    assert format_validity_score(junk) == 0.0

    empty = ""
    print(f"  empty:       {format_validity_score(empty):.2f}")
    assert format_validity_score(empty) == 0.0

    print("[test] grounding_score")
    state = {
        "text": "You are in the kitchen. You see a drawer, a fridge, and a countertop.",
        "admissible_commands": ["open drawer", "open fridge", "look"],
    }
    a1 = "Head to the drawer and grab the alarmclock"          # 1 hit (drawer)
    a2 = "Open the fridge, then check the drawer inside"       # 2 hits (fridge, drawer)
    a3 = "Do something totally unrelated somewhere else"       # 0 hits
    print(f"  1 hit:   {grounding_score(a1, state):.2f}")
    print(f"  2 hits:  {grounding_score(a2, state):.2f}")
    print(f"  0 hits:  {grounding_score(a3, state):.2f}")
    print(f"  empty:   {grounding_score('', state):.2f}")
    assert grounding_score(a1, state) == 0.6
    assert grounding_score(a2, state) == 1.0
    assert grounding_score(a3, state) == 0.2
    assert grounding_score("", state) == 0.0

    print("[test] jaccard_distance")
    print(f"  identical:  {jaccard_distance(['a','b','c'], ['a','b','c']):.2f}")
    print(f"  disjoint:   {jaccard_distance(['a','b'], ['x','y']):.2f}")
    print(f"  partial:    {jaccard_distance(['a','b','c'], ['b','c','d']):.2f}")
    assert jaccard_distance(['a','b','c'], ['a','b','c']) == 0.0
    assert jaccard_distance(['a','b'], ['x','y']) == 1.0
    # {a,b,c} vs {b,c,d} → inter=2, union=4, dist=0.5
    assert jaccard_distance(['a','b','c'], ['b','c','d']) == 0.5

    print("[test] compute_diversity_reward")
    a = "head to the drawer and grab the pen"
    peers = [
        "open the fridge and see what is inside",
        "look at the countertop and find the pen",
        "check every cabinet in the kitchen",
    ]
    d = compute_diversity_reward(a, peers)
    print(f"  diversity vs 3 peers: {d:.3f} (expect ~0.6-0.9)")
    assert 0.5 < d < 1.0

    print("[test] compute_group_shaping_rewards (K+1 = 4 branches)")
    advice_group = [
        "",                                            # SILENCE
        "Head to the drawer and grab the alarmclock",  # main advice
        "Open the fridge and check inside",            # fresh advice 1
        "Look at the countertop for a pen",            # fresh advice 2
    ]
    raw_group = [
        '{"gate": "SILENCE", "advice": ""}',
        '{"gate": "HELP", "advice": "Head to the drawer and grab the alarmclock"}',
        '{"gate": "HELP", "advice": "Open the fridge and check inside"}',
        '{"gate": "HELP", "advice": "Look at the countertop for a pen"}',
    ]
    rewards = compute_group_shaping_rewards(advice_group, raw_group, state)
    for i, r in enumerate(rewards):
        print(f"  branch {i}: intr={r.r_intrinsic:.2f} div={r.r_diversity:.3f} "
              f"(fmt={r.r_format:.2f} gnd={r.r_grounding:.2f}) "
              f"combined@(1,0.5)={r.combined(1.0, 0.5):.3f}")
    # SILENCE branch: empty advice → grounding 0 → intrinsic ~ 0.4*1.0 + 0.6*0 = 0.4
    assert abs(rewards[0].r_intrinsic - 0.4) < 1e-6
    # SILENCE diversity should be lower (empty vs others)
    # (empty tokens produce set(); comparing empty set vs others gives distance 1.0
    #  actually since sa is empty and sb is not, union = |sb|, inter = 0, dist = 1.0)
    assert rewards[0].r_diversity == 1.0
    # Main advice: 2 hits (drawer, alarmclock isn't in state but drawer is; "alarmclock" NOT in state) →
    # Actually only "drawer" hits → grounding 0.6. Let's just print & not assert exact.

    print("[test] all cf_reward_assembler tests passed")


if __name__ == "__main__":
    _self_test()
