"""
cf_token_masks.py — locate advice/gate value tokens in JSON-formatted output.

Companion emits assistant text like:
    {"gate": "HELP", "advice": "Head to the dresser and grab the alarmclock."}

For PPO clip loss we need to know which token positions belong to:
  - "gate value" tokens: the actual HELP or SILENCE literal (train by gate loss)
  - "advice value" tokens: the string content of `advice` (train by content loss)
  - everything else (JSON structure, keys, whitespace): masked out

If we don't mask correctly, content reward contaminates gate token gradient
(scheduler learns "when advice looks good, choose HELP" — wrong causal signal).

Implementation strategy: after tokenizing the full assistant text, we
re-tokenize just the value substrings and locate them in the full stream.
For robust matching we fall back to text-position search on the decoded
string when direct substring token match fails.

Design constraint: token boundaries in Qwen JSON output are unstable —
sometimes `"advice": "Head"` tokenizes as [`"`, `advice`, `":`, ` "`, `Head`],
sometimes as [`"advice"`, `:`, ` "Head`]. We handle both by decoding each
token range and matching text spans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TokenMask:
    """Boolean-like mask over the assistant response's token stream.

    Length equals the number of response tokens (not including prompt).
    """
    length: int
    gate_positions: List[int]      # indices of gate value tokens (e.g. HELP token)
    advice_positions: List[int]    # indices of advice value tokens

    def gate_mask(self) -> List[int]:
        m = [0] * self.length
        for i in self.gate_positions:
            if 0 <= i < self.length:
                m[i] = 1
        return m

    def advice_mask(self) -> List[int]:
        m = [0] * self.length
        for i in self.advice_positions:
            if 0 <= i < self.length:
                m[i] = 1
        return m

    def __repr__(self) -> str:
        return (f"TokenMask(len={self.length}, "
                f"gate={len(self.gate_positions)} pos, "
                f"advice={len(self.advice_positions)} pos)")


# ---------------------------------------------------------------------------
# JSON parsing (extract gate string and advice string cleanly)
# ---------------------------------------------------------------------------

_GATE_RE = re.compile(r'"gate"\s*:\s*"([^"]+)"')
_ADVICE_RE = re.compile(r'"advice"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def parse_companion_json(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (gate_str, advice_str) from companion output.

    Robust to whitespace, code-fence wrapping. Returns (None, None) if
    unable to parse.
    """
    s = text.strip()
    # Strip leading ```json / trailing ``` if any
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    g = _GATE_RE.search(s)
    a = _ADVICE_RE.search(s)
    gate = g.group(1).strip() if g else None
    advice = a.group(1) if a else None
    if advice is not None:
        # Un-escape common backslash sequences the model may emit
        advice = advice.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
    return gate, advice


# ---------------------------------------------------------------------------
# Token-level mask construction
# ---------------------------------------------------------------------------

def build_token_mask(
    tokenizer,
    response_text: str,
    response_ids: List[int],
) -> TokenMask:
    """Given companion response text + its tokenization, return TokenMask.

    Strategy:
      1. Parse response_text to extract gate string and advice string.
      2. Decode response_ids sequentially, tracking each token's byte span
         in the reconstructed text.
      3. For each character in the gate value substring, find which token
         it falls in; collect those token indices.
      4. Same for advice value.

    This is O(len(response_ids)) and robust to sub-word tokenization
    inconsistencies.
    """
    gate_str, advice_str = parse_companion_json(response_text)

    # Build per-token char spans by re-decoding token by token
    # Note: tokenizer.decode(single_token) doesn't always concatenate perfectly
    # so we accumulate: decode(ids[:i+1]) − decode(ids[:i]) = the i-th token's added chars
    prev_text = ""
    token_spans: List[Tuple[int, int]] = []   # (start_char, end_char) exclusive
    for i in range(len(response_ids)):
        cur_text = tokenizer.decode(response_ids[:i + 1], skip_special_tokens=False)
        added = cur_text[len(prev_text):]
        start_char = len(prev_text)
        end_char = start_char + len(added)
        token_spans.append((start_char, end_char))
        prev_text = cur_text
    full_decoded = prev_text

    gate_positions: List[int] = []
    advice_positions: List[int] = []

    if gate_str:
        for c_start, c_end in _find_all_substr_char_ranges(full_decoded, gate_str):
            gate_positions.extend(_token_indices_covering(token_spans, c_start, c_end))

    if advice_str:
        for c_start, c_end in _find_all_substr_char_ranges(full_decoded, advice_str):
            advice_positions.extend(_token_indices_covering(token_spans, c_start, c_end))

    # Deduplicate preserving order
    gate_positions = sorted(set(gate_positions))
    advice_positions = sorted(set(advice_positions))

    return TokenMask(
        length=len(response_ids),
        gate_positions=gate_positions,
        advice_positions=advice_positions,
    )


def _find_all_substr_char_ranges(haystack: str, needle: str) -> List[Tuple[int, int]]:
    """Find every occurrence of `needle` in `haystack`; return (start, end) char ranges.

    Uses simple str.find; case-sensitive. `end` is exclusive.
    Returns empty list if needle is empty.
    """
    if not needle:
        return []
    out: List[Tuple[int, int]] = []
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i < 0:
            break
        out.append((i, i + len(needle)))
        start = i + 1
    return out


def _token_indices_covering(
    token_spans: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> List[int]:
    """Return token indices whose char span overlaps [char_start, char_end)."""
    out = []
    for idx, (s, e) in enumerate(token_spans):
        # Token overlaps if not (e <= char_start or s >= char_end)
        if not (e <= char_start or s >= char_end):
            out.append(idx)
    return out


# ---------------------------------------------------------------------------
# Sanity self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Quick smoke: parse a mock output and verify masks."""
    # Simulate a tokenizer with word-level tokens (spaces as separators)
    class MockTok:
        def __init__(self, text):
            self.words = text.split(" ")
            # word_ids[i] = i (dummy id)

        def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
            return " ".join(self.words[i] for i in ids)

    text = '{ "gate" : "HELP" , "advice" : "grab the alarmclock" }'
    words = text.split(" ")
    tok = MockTok(text)
    ids = list(range(len(words)))

    mask = build_token_mask(tok, text, ids)
    print(f"[self-test] {mask}")
    print(f"  gate positions:  {mask.gate_positions}  (words: {[words[i] for i in mask.gate_positions]})")
    print(f"  advice positions: {mask.advice_positions} (words: {[words[i] for i in mask.advice_positions]})")

    # Parse test
    text2 = '{"gate": "SILENCE", "advice": "Head to the drawer, then grab the pen."}'
    g, a = parse_companion_json(text2)
    print(f"[self-test] parse: gate={g!r}, advice={a!r}")

    # Escape test
    text3 = '{"gate": "HELP", "advice": "Say \\"hello\\" to the player"}'
    g, a = parse_companion_json(text3)
    print(f"[self-test] parse w/ escape: gate={g!r}, advice={a!r}")


if __name__ == "__main__":
    _self_test()
