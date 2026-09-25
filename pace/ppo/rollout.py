"""
rollout.py — episode collection + buffer for PPO (SPARK track).

This is the SPARK-track rewrite of rl/ppo/rollout.py. Key differences:

  * The "state" is no longer a 17-dim hand-crafted score feature; it's a
    `features` dict that the new Policy consumes directly:
        {frames, task, state_compact, env_text, last_advice}
    We store the FULL dict (including PIL frames) on each Transition so
    the PPO update step can re-forward Policy on the exact same input.

  * `Policy.value(features)` replaces the old standalone ValueHead call
    (the value head is now part of Policy and conditions on the SPARK
    embeddings).

  * No ScoreFeatureBuilder dependency — there's no 17-dim feature to
    accumulate. SPARK embeddings are computed inside Policy each time.

`collect_episode` runs ONE full episode end-to-end:
  reset env → loop max_steps steps: build features → policy decide /
  generate_advice / value → env.step(advice) → assemble custom reward →
  append Transition. Returns an EpisodeRollout with bootstrap_value set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

from rl_spark.reward import RewardConfig, assemble_reward


# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    # The full features dict at decision time. Stored so the PPO update
    # step can re-forward Policy with the exact same input. `frames` is a
    # list of PIL.Image; the dict isn't a tensor so it lives outside of
    # FlatBatch and gets passed alongside as `transitions_meta` (a parallel
    # list).
    features: dict[str, Any]

    decision: str                  # "HELP" | "SILENCE"
    p_help: float                  # predicted p(HELP), for logging
    advice: str                    # generated advice text ("" if SILENCE)
    response_ids: list[int]        # token ids of advice (empty if SILENCE)
    logp_old: np.ndarray           # (R,) float32 per-token logp at rollout time

    value: float                   # V(s) at rollout time
    score: float                   # SparkScorer score AFTER this step
    reward: float                  # total assembled reward
    reward_log: dict[str, Any]     # reward breakdown for logging
    done: bool                     # terminal (e.g. agent died)
    truncated: bool                # truncated (max_steps reached)

    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeRollout:
    transitions: list[Transition]
    bootstrap_value: float = 0.0

    @property
    def length(self) -> int:
        return len(self.transitions)

    def stack(self, key: str) -> np.ndarray:
        return np.array([getattr(t, key) for t in self.transitions])

    def total_reward(self) -> float:
        return float(sum(t.reward for t in self.transitions))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _logp_of_response(
    policy,                                # rl_spark.policy.Policy
    features: dict,
    response_ids_t: torch.Tensor,          # (1, R) long on device
    task: str,
) -> np.ndarray:
    """Compute per-token logp of `response_ids_t` under the CURRENT policy.
    Used at rollout time to record logp_old. No grad."""
    if response_ids_t.shape[1] == 0:
        return np.zeros(0, dtype=np.float32)
    with torch.no_grad():
        logits, prompt_length = policy.forward_logits_with_response(
            features, response_ids_t, prompt_kind="advice", task=task,
        )
        Rt = response_ids_t.shape[1]
        pred = logits[:, prompt_length - 1 : prompt_length - 1 + Rt, :].float()
        logp = torch.log_softmax(pred, dim=-1)
        token_logp = logp.gather(-1, response_ids_t.unsqueeze(-1)).squeeze(-1)
    return token_logp[0].detach().cpu().numpy().astype(np.float32)


def _build_features(env, task: str, last_advice: Optional[str]) -> dict:
    """Pack the env's cached observation into the features dict Policy expects."""
    return {
        "frames": env.last_frames,
        "task": task,
        "state_compact": env.last_state_compact,
        "env_text": env.last_clip_text,
        "last_advice": last_advice,
    }


_DECISION_WORDS = ("yes", "Yes", "YES", "no", "No", "NO")


def _strip_decision_prefix(text: str) -> str:
    """Remove the leading 'yes'/'no' label the policy emits before its
    advice text. The label is useful for our PPO decision logic but the
    bot only needs the actionable suggestion that follows.

    A label is recognized iff:
      * it's one of the case variants in _DECISION_WORDS, AND
      * it's followed by a word boundary (space, colon, comma, period,
        newline) or end-of-string — so we don't accidentally strip
        'yes' from 'yesterday' or 'no' from 'noon'.

    Examples:
        'yes Chop the oak tree'   -> 'Chop the oak tree'
        'yes: chop the oak tree'  -> 'chop the oak tree'
        'no'                      -> ''
        'No, the agent is fine'   -> 'the agent is fine'
        'craft a stone pickaxe'   -> 'craft a stone pickaxe'  (unchanged)
    """
    if not text:
        return text
    s = text.lstrip()
    for word in _DECISION_WORDS:
        if s.startswith(word):
            rest = s[len(word):]
            # Require word boundary so 'yesterday' doesn't get butchered.
            if not rest or rest[0] in " :,.;\n\t":
                return rest.lstrip(" :,.;\n\t")
    return text


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def collect_episode(
    env,                  # rl_spark.mindcraft_env.MindcraftEnv
    policy,               # rl_spark.policy.Policy
    value_head=None,      # kept for API compat; actually use policy.value()
    *,
    task: str,
    task_spec=None,       # rl_spark.task_pool.TaskSpec | None
    reward_cfg: Optional[RewardConfig] = None,
    max_steps: Optional[int] = None,
    silence_advice: str = "",
) -> EpisodeRollout:
    """
    Run one episode end-to-end and produce an EpisodeRollout.

    If `task_spec` is provided, milestone-based reward shaping is active;
    the episode tracks `seen_milestones` so each milestone fires at most
    once. `task` (the natural-language string) is still passed separately
    because it's used by the policy prompt and SparkScorer regardless of
    whether milestone rewards are enabled.

    Side effects:
      * calls env.reset() at the start (caller doesn't need to)

    Returns an EpisodeRollout with bootstrap_value already set.
    """
    reward_cfg = reward_cfg or RewardConfig()

    obs, info = env.reset()
    initial_score = float(info.get("initial_score", 0.5))
    score_prev = initial_score
    last_advice: Optional[str] = None

    # Milestone tracking: which subtask labels have already fired this
    # episode. Persists across all steps; cleared by virtue of being a
    # local variable scoped to this episode.
    seen_milestones: set = set()

    transitions: list[Transition] = []
    step_idx = 0
    max_steps = max_steps or getattr(env, "max_steps", 30)

    while step_idx < max_steps:
        features = _build_features(env, task, last_advice)

        # 1. decide
        decision, p_help = policy.decide(features)

        # 2. value (no grad at rollout time)
        with torch.no_grad():
            v = policy.value(features)
            value = float(v.item())

        # 3. advice (only if HELP)
        if decision == "HELP":
            advice, gen_info = policy.generate_advice(features, task=task)
            response_ids = list(gen_info["new_ids"])
        else:
            advice = silence_advice
            response_ids = []

        # 4. log-prob of the response under the rollout policy
        if response_ids:
            response_t = torch.tensor(
                response_ids, device=policy.device, dtype=torch.long,
            ).unsqueeze(0)
            logp_old = _logp_of_response(policy, features, response_t, task)
        else:
            logp_old = np.zeros(0, dtype=np.float32)

        # 5. step env. Strip the "HELP"/"SILENCE" decision prefix before
        # sending — bot doesn't need to see the meta-label, only the
        # actionable suggestion. (The full text with prefix is kept on
        # Transition.advice for PPO logging.)
        bot_advice = _strip_decision_prefix(advice)
        log.info(
            f"  [rollout step] decision={decision} p_help={p_help:.3f} "
            f"raw_advice={advice!r} bot_advice={bot_advice!r}"
        )
        # state BEFORE action (already in features) and AFTER action
        # (in step_info) — both passed to assemble_reward so the
        # task_progress component can compute the inventory delta.
        state_prev = features.get("state_compact")
        obs2, env_reward, done, truncated, step_info = env.step(bot_advice)
        score_now = float(step_info.get("score", score_prev))
        state_now = step_info.get("state_compact")

        # 6. assemble custom reward (milestone tracking mutates seen_milestones)
        reward, reward_log = assemble_reward(
            score_now=score_now,
            score_prev=score_prev,
            decision=decision,
            p_help=p_help,
            advice_text=advice,
            cfg=reward_cfg,
            state_now=state_now,
            state_prev=state_prev,
            task_spec=task_spec,
            seen_milestones=seen_milestones,
        )

        # Diagnostic: compact inventory snapshot + which ms fired this
        # step. Lets us tell apart "bot got items but milestone didn't
        # fire" (reward bug) vs "bot got nothing in this step" (env /
        # advisee behavior issue).
        if isinstance(state_now, dict):
            inv = state_now.get("inventory_top") or []
            held = state_now.get("held_item") or {}
            inv_brief = [
                f"{it.get('name')}x{it.get('count')}"
                for it in inv if isinstance(it, dict)
                and it.get("count", 0) and it.get("name")
            ][:6]
            held_str = (
                f"{held.get('name')}x{held.get('count')}"
                if isinstance(held, dict) and held.get("name") else "none"
            )
        else:
            inv_brief, held_str = [], "n/a"
        ms_fired_this_step = reward_log.get("r/ms_fired", [])
        log.info(
            f"    [diag] held={held_str} inv={inv_brief} "
            f"ms_fired={ms_fired_this_step} step_reward={reward:.3f}"
        )

        transitions.append(
            Transition(
                features=features,
                decision=decision,
                p_help=p_help,
                advice=advice,
                response_ids=response_ids,
                logp_old=logp_old,
                value=value,
                score=score_now,
                reward=reward,
                reward_log=reward_log,
                done=bool(done),
                truncated=bool(truncated),
                info=step_info,
            )
        )

        score_prev = score_now
        last_advice = advice if decision == "HELP" else last_advice
        step_idx += 1
        if done or truncated:
            break

    # Bootstrap value: 0 if terminated, V(s_T) if truncated
    bootstrap_value = 0.0
    if transitions and transitions[-1].truncated and not transitions[-1].done:
        post_features = _build_features(env, task, last_advice)
        with torch.no_grad():
            bootstrap_value = float(policy.value(post_features).item())

    return EpisodeRollout(transitions=transitions, bootstrap_value=bootstrap_value)


# ---------------------------------------------------------------------------
# flat buffer for mini-batching
# ---------------------------------------------------------------------------

@dataclass
class FlatBatch:
    """One mini-batch fed to the PPO update step.

    Tensor fields are on the policy's device. `features_list` carries the
    per-sample features dicts (with PIL frames inside) — these are NOT
    tensors so they live in a Python list alongside.

    response_ids / logp_olds are variable-length per sample; we forward
    one sample at a time during update, so no padding-mask plumbing.
    """
    features_list: list[dict]              # length B, each a features dict
    response_ids: list[torch.Tensor]       # list[(R_i,) long]
    logp_olds: list[torch.Tensor]          # list[(R_i,) float]
    advantages: torch.Tensor               # (B,)
    returns: torch.Tensor                  # (B,)
    values_old: torch.Tensor               # (B,)
    is_help: torch.Tensor                  # (B,) bool

    @property
    def size(self) -> int:
        return self.advantages.shape[0]


def flatten_episodes(
    episodes: list[EpisodeRollout],
    advantages_per_ep: list[np.ndarray],
    returns_per_ep: list[np.ndarray],
    device: torch.device,
) -> FlatBatch:
    """Stack all transitions across episodes into one flat batch."""
    features_list, response_ids, logp_olds = [], [], []
    advs, rets, vals, is_help = [], [], [], []

    for ep, A, R in zip(episodes, advantages_per_ep, returns_per_ep):
        for i, t in enumerate(ep.transitions):
            features_list.append(t.features)
            advs.append(float(A[i]))
            rets.append(float(R[i]))
            vals.append(float(t.value))
            response_ids.append(
                torch.tensor(t.response_ids, dtype=torch.long, device=device)
            )
            logp_olds.append(
                torch.tensor(t.logp_old, dtype=torch.float32, device=device)
            )
            is_help.append(t.decision == "HELP")

    return FlatBatch(
        features_list=features_list,
        response_ids=response_ids,
        logp_olds=logp_olds,
        advantages=torch.tensor(advs, dtype=torch.float32, device=device),
        returns=torch.tensor(rets, dtype=torch.float32, device=device),
        values_old=torch.tensor(vals, dtype=torch.float32, device=device),
        is_help=torch.tensor(is_help, dtype=torch.bool, device=device),
    )
