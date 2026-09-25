#!/usr/bin/env bash
# =====================================================================
# MC NIC training pipeline (VL companion).
# Staged (matches the method):  base VL (warmup) -> gate_only -> content_only -> joint
# Each phase resumes from the previous phase's final checkpoint and auto-evals.
#
# Run phase by phase (recommended, so you can check results between),
# or all at once. Requires: bot + reverse tunnel (:18765) + vLLM 8000/8001.
# =====================================================================
set -uo pipefail
cd /workspace/credit && export PYTHONPATH=. PYTHONUNBUFFERED=1

SPARK=http://172.17.0.1:18765
BASE=/workspace/models/Qwen3.5-9B
ADV=/workspace/models/Qwen3.5-4B
CK=/workspace/checkpoints
N_UPD=10                       # updates per phase
CKN=$(printf "mc_ckpt_step_%04d" $N_UPD)   # last ckpt name of each phase

# shared args for every phase
COMMON="--base-model $BASE --spark-url $SPARK \
  --advisee-url http://localhost:8000 --advisee-model $ADV \
  --companion-url http://localhost:8001 --companion-model companion \
  --n-updates $N_UPD --n-episodes 2 --K 2 --branch-max-steps 6 --max-help-states 4 \
  --min-step-seconds 5 --eval-episodes 10"

run() {  # run <name> <init-args> <train-mode>
  local name=$1 init=$2 mode=$3
  local out=$CK/$name
  local log=/workspace/logs/${name}_$(date +%m%d_%H%M).log
  echo "==================== PHASE $name ($mode) ===================="
  python3 -u rl_causal/scripts/train_mc.py $COMMON \
    $init --train-mode $mode --output-dir $out 2>&1 | tee "$log"
}

# --- preflight ---
curl -sf "$SPARK/health" >/dev/null && echo "[ok] bot reachable" || { echo "[FAIL] tunnel down: $SPARK"; exit 1; }
curl -sf localhost:8000/v1/models >/dev/null && echo "[ok] advisee 8000" || { echo "[FAIL] 8000"; exit 1; }
curl -sf localhost:8001/v1/models >/dev/null && echo "[ok] companion 8001" || { echo "[FAIL] 8001"; exit 1; }

# --- Phase 1: gate_only (fresh base VL; content stays = base VL grounding) ---
run mc_vl_gate "--create-lora" gate_only

# --- Phase 2: content_only (resume from gate phase) ---
run mc_vl_content \
  "--adapter-path $CK/mc_vl_gate/$CKN --value-head-path $CK/mc_vl_gate/$CKN/value_head.pt" \
  content_only

# --- Phase 3: joint (resume from content phase) ---
run mc_vl_joint \
  "--adapter-path $CK/mc_vl_content/$CKN --value-head-path $CK/mc_vl_content/$CKN/value_head.pt" \
  joint

echo "==================== PIPELINE DONE ===================="
echo "final companion: $CK/mc_vl_joint/$CKN"
