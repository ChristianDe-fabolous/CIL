#!/bin/bash
# ── ETH ISG Student Cluster — Action Phase VLM Benchmark ─────────────────────
#
# Usage:
#   sbatch scripts/run_slurm.sh
#
# Override defaults via env vars:
#   MODEL=qwen-7b-int8 sbatch scripts/run_slurm.sh
#   DATASET=data/action_phase_dataset_singleview.jsonl sbatch scripts/run_slurm.sh
#   COT=1 sbatch scripts/run_slurm.sh
#
# Resume a previous run:
#   RUN_ID=my_run sbatch scripts/run_slurm.sh
#
# Smoke test (first 10 questions):
#   SMOKE=1 sbatch scripts/run_slurm.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=vlm-bench
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=cdeubel@ethz.ch
#SBATCH --mail-type=END,FAIL

# ── GPU / model reference ─────────────────────────────────────────────────────
# qwen-3b        → 1080ti  (~6GB VRAM)   --gpus=1080ti:1
# qwen-7b-int8   → 2080ti  (~8GB VRAM)   --gpus=2080ti:1
# qwen-7b        → 5060ti  (~14GB VRAM)  --gpus=5060ti:1  ← default
# qwen-32b-int8  → gb10    (~32GB VRAM)  --gpus=gb10:1    --time=24:00:00
# ─────────────────────────────────────────────────────────────────────────────

REPO=/work/courses/3dv/team29/3D-vision-Benchmarking-Spatial-and-State-Reasoning-Skills-of-VLMs-for-Robotics

# ── Defaults (override via env vars at sbatch time) ───────────────────────────
MODEL="${MODEL:-qwen-7b-int8}"
DATASET="${DATASET:-data/action_phase_dataset.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# ── Environment ───────────────────────────────────────────────────────────────
module load cuda/13.0
source ~/.bashrc   # activates conda env + sets HF_HOME, TRANSFORMERS_CACHE, etc.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Job info ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "Model   : $MODEL"
echo "Dataset : $DATASET"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO"

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python src/main.py \
    --task action_phase \
    --model $MODEL \
    --action-phase-data $DATASET \
    --batch-size $BATCH_SIZE \
    --cot"


[ -n "$ACTION_PHASE_TYPE" ] && CMD="$CMD --action-phase-type $ACTION_PHASE_TYPE"
[ -n "$RUN_ID" ]            && CMD="$CMD --run-id $RUN_ID --resume"
[ -n "$LIMIT" ]             && CMD="$CMD --limit $LIMIT"
[ "${SMOKE:-0}" = "1" ]     && CMD="$CMD --limit 10 --describe"

echo "CMD: $CMD"
eval $CMD

echo "========================================"
echo "Job $SLURM_JOB_ID done."
