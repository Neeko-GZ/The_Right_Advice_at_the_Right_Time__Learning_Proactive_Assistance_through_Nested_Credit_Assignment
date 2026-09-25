"""
MindcraftEnv (SPARK track) — gymnasium env that drives a remote Mindcraft bot
through SparkAPI, with two additions vs rl/mindcraft_env.py:

  1. /frames now returns `clip_text` — the per-window natural-language
     description of what the bot is doing (set externally via /clip_text).
     We surface it on every fetch so the SPARK scorer can use it as env_text
     and the policy can include it in the LM prompt.

  2. /state_compact — slim observation tailored for the VLM prompt
     (task + held + top inv + nearby + vitals). Avoids feeding the policy
     the full /state (which has 24 nearby entities, 36 inventory slots,
     velocity, exp, etc. — way too much).

The Scorer protocol is unchanged: `scorer(frames) -> float`. The SPARK
scorer wrapper (rl_spark/spark_scorer.py) binds task and env_text into
itself and exposes the same callable surface, so the env stays agnostic.

Topology (unchanged):
    PPO trainer (this process)
        ↓ reset() / step(advice_text)
    MindcraftEnv  ─────HTTP─────► SparkAPI (Windows, :8765 via SSH tunnel)
        │                             │
        ├─ SPARK (GPU, local)         ├─ /advice
        └─ reward = Δscore            ├─ /frames?n=16   (+ clip_text)
                                      ├─ /state_compact
                                      ├─ /execute_command
                                      └─ /reset
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import requests
from PIL import Image

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

log = logging.getLogger(__name__)


# ----------------------------- scorer protocol ------------------------------

Scorer = Callable[[list[Image.Image]], float]


class DummyScorer:
    """Smoke-test scorer: returns a random walk, not a real model."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.last = 0.5

    def __call__(self, frames: list[Image.Image]) -> float:
        self.last = float(np.clip(self.last + self.rng.normal(0, 0.05), 0.0, 1.0))
        return self.last


# ----------------------------- SparkAPI client ------------------------------

@dataclass
class FramesResult:
    """Bundle returned by SparkClient.frames(). The clip_text may be None
    if no /clip_text POST has happened yet for this episode."""
    frames: list[Image.Image]
    clip_text: Optional[str]
    clip_text_ts: Optional[int]


def _build_retry_session(total_retries: int = 5, backoff_factor: float = 0.5) -> requests.Session:
    """Session with auto-retry on transient network errors (connection
    reset, remote end disconnect) — common when SSH tunnel hiccups or the
    Mindcraft Node.js side momentarily stalls.

    Retries on:
      * Connection errors (incl. RemoteDisconnected via urllib3.exceptions.ProtocolError)
      * 502 / 503 / 504 status codes
    With exponential backoff: 0.5s, 1s, 2s, 4s, 8s = ~15s total worst case.
    """
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter

    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess = requests.Session()
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


@dataclass
class SparkClient:
    base_url: str = "http://localhost:8765"
    timeout: float = 10.0
    _session: Optional[requests.Session] = None

    def __post_init__(self):
        if self._session is None:
            self._session = _build_retry_session()

    def _get(self, path: str, params: Optional[dict] = None, timeout: Optional[float] = None):
        url = f"{self.base_url}{path}"
        try:
            r = self._session.get(url, params=params, timeout=timeout or self.timeout)
        except requests.exceptions.ConnectionError as e:
            log.error(
                f"  HTTP GET {url} failed after retries: {e!r}. "
                f"Check Windows: (1) mindcraft node process alive? "
                f"(2) curl http://localhost:8765/health from Linux works? "
                f"(3) SSH tunnel still up?"
            )
            raise
        r.raise_for_status()
        return r

    def _post(self, path: str, json: dict, timeout: Optional[float] = None):
        url = f"{self.base_url}{path}"
        try:
            r = self._session.post(url, json=json, timeout=timeout or self.timeout)
        except requests.exceptions.ConnectionError as e:
            log.error(
                f"  HTTP POST {url} failed after retries: {e!r}. "
                f"Check Windows mindcraft + SSH tunnel."
            )
            raise
        r.raise_for_status()
        return r

    def health(self) -> dict:
        return self._get("/health").json()

    def state(self) -> dict:
        return self._get("/state").json()

    def state_compact(self, top_inv: int = 8, top_near: int = 4) -> dict:
        return self._get(
            "/state_compact",
            params={"top_inv": top_inv, "top_near": top_near},
        ).json()

    def frames(self, n: int = 16) -> FramesResult:
        payload = self._get("/frames", params={"n": n}).json()
        imgs: list[Image.Image] = []
        for f in payload["frames"]:
            buf = base64.b64decode(f["jpeg_b64"])
            imgs.append(Image.open(io.BytesIO(buf)).convert("RGB"))
        return FramesResult(
            frames=imgs,
            clip_text=payload.get("clip_text"),
            clip_text_ts=payload.get("clip_text_ts"),
        )

    def advice(self, text: str, mode: str = "history") -> dict:
        return self._post("/advice", json={"advice": text, "mode": mode}).json()

    def execute_command(self, command: str, timeout: Optional[float] = None) -> dict:
        """Run a Mindcraft skill directly (e.g. '!digDown(3)'), bypassing
        bot.chat() / LLM. Used by the scripted-bot data collector — Mindcraft
        filters self-chat so /advice mode='chat' would not trigger commands
        when posted by the bot itself.

        `timeout` (sec) defaults to a long wait because skills like
        collectBlocks can take 10s+ to finish.
        """
        return self._post(
            "/execute_command",
            json={"command": command},
            timeout=timeout if timeout is not None else max(self.timeout, 60.0),
        ).json()

    def set_clip_text(self, text: Optional[str]) -> dict:
        """Set per-window natural-language description (env_text supervisor).
        Pass None or '' to clear. Used by scripted-bot driver during SPARK
        pretraining; not called during PPO RL (clip_text is set by the bot
        side or left null)."""
        return self._post("/clip_text", json={"text": text}).json()

    def reset(self, **kwargs) -> dict:
        return self._post("/reset", json=kwargs).json()


# -------------------------------- env ---------------------------------------

class MindcraftEnv(gym.Env):
    """
    A single-bot Minecraft RL env (SPARK track). Each step:
        1. POST /advice with the companion's advice text
        2. sleep(step_seconds) while the survival agent acts
        3. GET /frames (N=16, +clip_text) and /state_compact
        4. reward = scorer(frames) - prev_score   (Δscore)

    Differences vs rl/MindcraftEnv:
      * info dict carries `clip_text` and `state_compact` so the policy
        can build its VLM prompt from a single source.
      * The frames result is structured (FramesResult) instead of a bare list.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scorer: Optional[Scorer] = None,
        base_url: str = "http://localhost:8765",
        step_seconds: float = 4.0,
        max_steps: int = 30,
        n_frames: int = 16,
        advice_mode: str = "history",
        reset_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.client = SparkClient(base_url=base_url)
        self.scorer = scorer or DummyScorer()
        self.step_seconds = step_seconds
        self.max_steps = max_steps
        self.n_frames = n_frames
        self.advice_mode = advice_mode
        self.reset_kwargs = reset_kwargs or {}

        # Action space: arbitrary text (advice). Strings flow through step().
        self.action_space = spaces.Text(max_length=2048)

        # Observation space is loose because the policy consumes it through
        # its own Qwen-VL preprocessing. We just publish a few scalars so the
        # env validates as a gym.Env.
        self.observation_space = spaces.Dict({
            "frames_shape": spaces.Box(low=0, high=10000, shape=(3,), dtype=np.int32),
            "health": spaces.Box(low=0, high=20, shape=(), dtype=np.float32),
            "food": spaces.Box(low=0, high=20, shape=(), dtype=np.float32),
        })

        self._step_count = 0
        self._prev_score: Optional[float] = None

        # Cached last fetch — accessed by the policy (which doesn't touch
        # the gym obs dict for VLM input, since the obs dict is too lossy).
        self._last_frames: list[Image.Image] = []
        self._last_clip_text: Optional[str] = None
        self._last_state_compact: dict = {}

    # ------------------------ public accessors -------------------------
    # The PPO/policy code reads these instead of the gym obs dict, because
    # the obs dict is intentionally tiny.

    @property
    def last_frames(self) -> list[Image.Image]:
        return self._last_frames

    @property
    def last_clip_text(self) -> Optional[str]:
        return self._last_clip_text

    @property
    def last_state_compact(self) -> dict:
        return self._last_state_compact

    # ------------------------ internals --------------------------------

    def _fetch_obs(self) -> dict:
        fr = self.client.frames(n=self.n_frames)
        sc = self.client.state_compact()
        self._last_frames = fr.frames
        self._last_clip_text = fr.clip_text
        self._last_state_compact = sc

        return {
            "frames_shape": np.array(
                [
                    len(fr.frames),
                    fr.frames[0].height if fr.frames else 0,
                    fr.frames[0].width if fr.frames else 0,
                ],
                dtype=np.int32,
            ),
            "health": np.float32(sc.get("health") or 0.0),
            "food": np.float32(sc.get("food") or 0.0),
        }

    def _score(self, frames: list[Image.Image]) -> float:
        return float(self.scorer(frames))

    # ------------------------ gym API ----------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._step_count = 0

        self._wait_for_spawn()
        self.client.reset(**(self.reset_kwargs | (options or {})))
        time.sleep(1.0)  # bot settle

        obs = self._fetch_obs()
        self._prev_score = self._score(self._last_frames)
        info = {
            "initial_score": self._prev_score,
            "state_compact": self._last_state_compact,
            "clip_text": self._last_clip_text,
        }
        return obs, info

    def step(self, action: str):
        assert isinstance(action, str), f"action must be a string, got {type(action)}"

        if action.strip():
            try:
                self.client.advice(action, mode=self.advice_mode)
            except Exception as e:
                log.warning("advice POST failed: %s", e)

        time.sleep(self.step_seconds)

        obs = self._fetch_obs()
        score = self._score(self._last_frames)
        reward = score - (self._prev_score if self._prev_score is not None else score)
        self._prev_score = score

        self._step_count += 1
        sc = self._last_state_compact
        dead = (sc.get("health") or 0) <= 0
        done = bool(dead)
        truncated = self._step_count >= self.max_steps

        info = {
            "score": score,
            "reward": reward,
            "state_compact": sc,
            "clip_text": self._last_clip_text,
            "dead": dead,
            "step": self._step_count,
        }
        return obs, float(reward), done, bool(truncated), info

    def close(self):
        pass

    # ------------------------ helpers ----------------------------------

    def _wait_for_spawn(self, timeout: float = 30.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                h = self.client.health()
                if h.get("spawned") and h.get("frame_recorder"):
                    return
            except Exception:
                pass
        raise RuntimeError("Bot did not spawn within timeout; check SparkAPI and Mindcraft log.")
