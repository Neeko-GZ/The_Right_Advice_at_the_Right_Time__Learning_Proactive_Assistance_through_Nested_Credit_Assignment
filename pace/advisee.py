"""
advisee.py — AdviseeInterface implementations for real inference.

Two classes:
  - VllmAdvisee    : calls a local vllm serve endpoint
  - OpenAIAdvisee  : calls OpenAI Chat Completions API

Both implement `AdviseeInterface.act(obs, advice, seed) → str` and satisfy
the PC1 seedable contract (same seed → same output) as long as the underlying
backend does. The prompt structure is adapted from pc1_responsiveness.py v4:
JSON output with thoughts + action, plus a `Companion advice:` field when
`advice is not None` (this is what makes brancher HELP branches differ from
SILENCE).

The advice field is the *only* structural difference from pc1_responsiveness's
prompt. Everything else (task, target block, history, current obs, inventory,
admissible, few-shot, JSON schema) matches — this keeps PC1 baseline numbers
comparable to brancher's SILENCE-branch numbers.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# prompt template — imported from rl_causal.prompts.alfworld for env portability
# (Phase 2 MC will swap to rl_causal.prompts.mc)
# ---------------------------------------------------------------------------

from rl_causal.prompts.alfworld import (
    SYSTEM_PROMPT_ADVISEE as _SYSTEM_PROMPT,
    FEW_SHOT_ADVISEE as _FEW_SHOT,
    build_advisee_prompt as _build_advisee_prompt,
)
from rl_causal.prompts.base import summarize_obs as _summarize_obs


# ---------------------------------------------------------------------------
# Per-env prompt bundle. ALFWorld = text actions validated against
# admissible_commands; Minecraft = free-form mindcraft skill calls (!cmd(...))
# with no admissible list, so parsing extracts the skill call directly.
# ---------------------------------------------------------------------------

def _prompt_bundle(env: str):
    if env in ("minecraft", "mc"):
        from rl_causal.prompts import mc as _mcp
        return (_mcp.SYSTEM_PROMPT_ADVISEE, _mcp.FEW_SHOT_ADVISEE,
                _mcp.build_advisee_prompt)
    return (_SYSTEM_PROMPT, _FEW_SHOT, _build_advisee_prompt)


_MC_CMD_RE = re.compile(r'!\s*[A-Za-z]\w*\s*\([^)]*\)')


def _extract_mc_command(raw: str) -> str:
    """Pull the mindcraft skill call `!name(args)` from the model output.

    Advisee replies as {"thoughts": "...", "action": "!skill(args)"}. We prefer
    the parsed `action` field (JSON unescapes the inner quotes correctly), then
    fall back to a raw regex over the whole text.
    """
    if not raw:
        return ""
    src = raw
    i, j = raw.find("{"), raw.rfind("}")
    if i != -1 and j > i:
        try:
            obj = json.loads(raw[i:j + 1])
            if isinstance(obj, dict) and obj.get("action"):
                src = str(obj["action"])
        except Exception:
            m = re.search(r'"action"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
            if m:
                try:
                    src = m.group(1).encode().decode("unicode_escape")
                except Exception:
                    src = m.group(1)
    m = _MC_CMD_RE.search(src)
    if m:
        cmd = m.group(0)
    else:
        cmd = src.strip()
        for line in src.splitlines():
            line = line.strip()
            if line.startswith("!"):
                cmd = line
                break
    # Strip stray JSON escaping that leaked through any path: some players
    # (e.g. Andy) emit !smeltItem(\"raw_iron\", 3) with literal backslash-quotes,
    # which mindcraft then parses as 0 args. Skill args never contain a real
    # backslash-escape, so unescaping quotes here is safe.
    return cmd.replace("! ", "!").replace('\\"', '"').replace("\\'", "'").strip()


# ---------------------------------------------------------------------------
# action extraction / validation
# ---------------------------------------------------------------------------

_JSON_ACTION_RE = re.compile(r'"action"\s*:\s*"([^"]+)"', re.IGNORECASE)
_ACTION_LINE_RE = re.compile(r"(?:^|\n)\s*action\s*:\s*(.+?)\s*(?:\n|$)", re.IGNORECASE)


def _extract_action_text(raw: str) -> str:
    if not raw:
        return ""
    m = _JSON_ACTION_RE.search(raw)
    if m:
        return m.group(1).strip().strip('"').strip("'").rstrip(".")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S | re.I)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict) and "action" in obj:
                return str(obj["action"]).strip().rstrip(".")
        except Exception:
            pass
    stripped = raw.strip().lstrip("﻿")
    i, j = stripped.find("{"), stripped.rfind("}")
    if i != -1 and j > i:
        try:
            obj = json.loads(stripped[i:j+1])
            if isinstance(obj, dict) and "action" in obj:
                return str(obj["action"]).strip().rstrip(".")
        except Exception:
            pass
    m = _ACTION_LINE_RE.search(raw)
    if m:
        return m.group(1).strip().strip('"').strip("'").rstrip(".")
    return raw.strip().strip('"').strip("'")


def _validate_action(raw: str, admissible: list) -> tuple:
    """Return (action_to_execute, is_valid). Same cascade as pc1_responsiveness."""
    if not admissible:
        return raw, False
    r = _extract_action_text(raw)
    r_lower = r.lower()
    if r in admissible:
        return r, True
    lower_map = {a.lower(): a for a in admissible}
    if r_lower in lower_map:
        return lower_map[r_lower], True
    for a in admissible:
        al = a.lower()
        if r_lower == al or r_lower.startswith(al + " ") or r_lower.startswith(al + "."):
            return a, True
    return admissible[0], False


# ---------------------------------------------------------------------------
# Vllm advisee
# ---------------------------------------------------------------------------

def _build_messages(user_prompt: str, system: str = None, few_shot: list = None) -> list:
    system = system if system is not None else _SYSTEM_PROMPT
    few_shot = few_shot if few_shot is not None else _FEW_SHOT
    msgs = [{"role": "system", "content": system}]
    for ex in few_shot:
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


class VllmAdvisee:
    """Advisee backed by a local vllm serve endpoint.

    Implements the AdviseeInterface Protocol from brancher.py.
    """

    def __init__(
        self,
        model: str,
        url: str = "http://localhost:8000",
        max_tokens: int = 200,
        temperature: float = 0.3,
        top_p: float = 1.0,
        top_k: int = -1,               # -1 = disabled (vLLM)
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        timeout: float = 60.0,
        history_window: int = 8,
        env: str = "alfworld",   # "alfworld" | "minecraft"
        vision: bool = False,    # send obs frames as image_url (VL players)
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.timeout = timeout
        self.history_window = history_window
        self.env = env
        self.vision = vision
        self._system, self._few_shot, self._build_prompt = _prompt_bundle(env)
        # Per-episode history. Each entry = (action_at_step_i, obs_text_AFTER_step_i).
        # `_pending_action` holds the action from last act() call whose result
        # (post-step obs) we haven't seen yet — recorded on the next act() call
        # when we get the fresh obs. This matches pc1_responsiveness semantics.
        self._history: list = []
        self._pending_action = None

    def reset_history(self) -> None:
        self._history = []
        self._pending_action = None

    def act(self, obs: dict, advice, seed: int) -> str:
        """Advisee decision. `obs` must have task_description / text /
        inventory_text / admissible_commands. `advice` is None for SILENCE
        branches, a string for HELP branches."""
        # Step 1: If we have an unclosed action from last call, close it now
        # by pairing it with the CURRENT obs (which is the result of that action).
        if self._pending_action is not None:
            self._history.append((self._pending_action, obs.get("text", "")))
            self._pending_action = None

        # Step 2: Build prompt with correctly-formed history (env-specific)
        prompt = self._build_prompt(
            obs=obs, advice=advice,
            history=self._history, history_window=self.history_window,
        )
        # VL players: attach the player's view frame(s) as image_url (same
        # mechanism the VL companion uses in prompts.mc.build_companion_prompt).
        if self.vision:
            frames = obs.get("frames_b64") or []
            if frames:
                prompt = [{"type": "text", "text": prompt}] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
                    for b in frames
                ]
        payload = {
            "model": self.model,
            "messages": _build_messages(prompt, self._system, self._few_shot),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": seed,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = requests.post(self.url + "/v1/chat/completions", json=payload, timeout=self.timeout)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()

        if self.env in ("minecraft", "mc"):
            # No admissible list in MC: extract the mindcraft skill call directly.
            action = _extract_mc_command(raw)
        else:
            admissible = obs.get("admissible_commands", []) or [""]
            action, _valid = _validate_action(raw, admissible)

        # Step 3: Remember this action; its result-obs will arrive next call.
        self._pending_action = action
        return action


# ---------------------------------------------------------------------------
# OpenAI advisee (gpt-4o-mini and friends)
# ---------------------------------------------------------------------------

class OpenAIAdvisee:
    """Advisee backed by OpenAI Chat Completions API. Same interface as
    VllmAdvisee. Requires OPENAI_API_KEY env var."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 200,
        temperature: float = 0.3,
        history_window: int = 8,
        base_url: str = None,            # e.g. https://api.deepseek.com for DeepSeek
        api_key_env: str = "OPENAI_API_KEY",
        env: str = "alfworld",           # "alfworld" | "minecraft"
        vision: bool = False,            # send obs frames as image_url (VL players)
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"{api_key_env} env var not set")
        _kw = {"api_key": api_key}
        if base_url:
            _kw["base_url"] = base_url
        self._client = OpenAI(**_kw)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history_window = history_window
        self.env = env
        self.vision = vision
        # env-specific system prompt / few-shot / prompt builder (same as VllmAdvisee)
        self._system, self._few_shot, self._build_prompt = _prompt_bundle(env)
        self._history: list = []
        self._pending_action = None

    def reset_history(self) -> None:
        self._history = []
        self._pending_action = None

    def act(self, obs: dict, advice, seed: int) -> str:
        # Close last pending action with the current (post-step) obs
        if self._pending_action is not None:
            self._history.append((self._pending_action, obs.get("text", "")))
            self._pending_action = None

        prompt = self._build_prompt(
            obs=obs, advice=advice,
            history=self._history, history_window=self.history_window,
        )
        # VL players: attach the player's view frame(s) as image_url, same
        # mechanism the VL companion uses (prompts.mc.build_companion_prompt).
        if self.vision:
            frames = obs.get("frames_b64") or []
            if frames:
                prompt = [{"type": "text", "text": prompt}] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
                    for b in frames
                ]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=_build_messages(prompt, self._system, self._few_shot),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=seed,
            top_p=1.0,
        )
        raw = resp.choices[0].message.content.strip()
        if self.env in ("minecraft", "mc"):
            action = _extract_mc_command(raw)
        else:
            admissible = obs.get("admissible_commands", []) or [""]
            action, _valid = _validate_action(raw, admissible)

        self._pending_action = action
        return action


# ---------------------------------------------------------------------------
# OpenAI-compatible COMPANION (DeepSeek / gpt-4o / any chat endpoint)
# ---------------------------------------------------------------------------

class OpenAICompanion:
    """Companion backed by an OpenAI-compatible chat API (gpt-4o, DeepSeek, ...).

    Same act() contract as OfflineCompanion / NullCompanion:
        act(obs, history, seed) -> (gate, advice, ok)

    Consumes the SAME companion prompt + milestones as the trained 9B companion
    (build_companion_prompt from prompts.alfworld), so all companion conditions
    see identical information — a fair head-to-head. gate/advice are parsed with
    the same robust regex used by OfflineCompanion (gate literal survives even if
    advice is truncated)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 200,
        temperature: float = 0.0,
        history_window: int = 10,
        base_url: str = None,            # https://api.deepseek.com for DeepSeek
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"{api_key_env} env var not set")
        _kw = {"api_key": api_key}
        if base_url:
            _kw["base_url"] = base_url
        self._client = OpenAI(**_kw)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.hw = history_window
        # A local vLLM endpoint (companion served on :8001) is a Qwen thinking
        # model; force no-think to match training + the VllmAdvisee path. Real
        # APIs (gpt-4o / DeepSeek) reject this kwarg, so only send it locally.
        self._local_vllm = bool(base_url) and (
            "localhost" in base_url or "127.0.0.1" in base_url)

    def act(self, obs, history, seed=None):
        from rl_causal.prompts.alfworld import (
            SYSTEM_PROMPT_COMPANION, build_companion_prompt,
        )
        user = build_companion_prompt(obs, history=history, history_window=self.hw)
        _kw = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_COMPANION},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=seed,
        )
        if self._local_vllm:
            _kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        resp = self._client.chat.completions.create(**_kw)
        gen = resp.choices[0].message.content or ""
        gm = re.search(r'"gate"\s*:\s*"([A-Za-z]+)"', gen)
        am = re.search(r'"advice"\s*:\s*"([^"]*)', gen)
        gate = gm.group(1).strip().upper() if gm else "SILENCE"
        advice = am.group(1) if am else ""
        ok = gm is not None
        return gate, advice, ok
