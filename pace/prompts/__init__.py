"""
rl_causal.prompts — env-specific prompt building.

Each env implementation provides a module (e.g. alfworld.py, mc.py) with:
  - system_prompt_advisee   : str
  - system_prompt_companion : str
  - few_shot_advisee        : list of {"user": str, "assistant": str}
  - build_advisee_prompt(obs, advice, history, ...) → str
  - build_companion_prompt(obs, history, ...) → str

This separation lets brancher / advisee / SFT training / PPO all be
env-agnostic — only prompt templates change per env. When adding a new
env (Phase 2 MC), write the equivalent module and swap the import.
"""
