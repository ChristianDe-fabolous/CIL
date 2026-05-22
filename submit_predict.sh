#!/bin/bash
# ── ETH ISG Student Cluster — CIL submission generation ──────────────────────
#
# Usage:
#   CHECKPOINT=/path/to/best.pth  TEST_DIR=/path/to/test/images  sbatch submit_predict.sh
#
# Optional overrides:
#   OUTPUT=/path/to/submission.csv   (default: submissions/<job_id>/submission.csv)
#   BATCH_SIZE=16                    (default: 8)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=CIL-predict
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --account=cil_jobs
#SBATCH --gpus=2080ti:1
#SBATCH --time=01:00:00
#SBATCH --mail-user=cdeubel@ethz.ch
#SBATCH --mail-type=END,FAIL

# ── Required (override via env vars) ─────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-/work/scratch/cdeubel/CIL/checkpoints/vit_depth_transformer_pretrainedTrue/best.pth}"
TEST_DIR="${TEST_DIR:-/cluster/courses/cil/monocular-depth-estimation/test/images}"

# ── Defaults ──────────────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-8}"
PRED_DIR="predictions/${SLURM_JOB_ID}"
OUTPUT="${OUTPUT:-submissions/${SLURM_JOB_ID}/submission.csv}"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO="/work/scratch/cdeubel/CIL"
SCRATCH_ENV="/work/scratch/cdeubel/CIL/.venv"

# ── Environment ───────────────────────────────────────────────────────────────
module load cuda/12.6.0
export UV_PROJECT_ENVIRONMENT="$SCRATCH_ENV"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ! command -v uv &>/dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$REPO"
mkdir -p logs predictions submissions

uv sync --frozen

# ── Job info ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Checkpoint : $CHECKPOINT"
echo "Test dir   : $TEST_DIR"
echo "Pred dir   : $PRED_DIR"
echo "Output CSV : $OUTPUT"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Inference ─────────────────────────────────────────────────────────────────
uv run python src/predict.py \
    --checkpoint "$CHECKPOINT" \
    --test_dir   "$TEST_DIR" \
    --output_dir "$PRED_DIR" \
    --batch_size "$BATCH_SIZE"

# ── Build submission CSV ───────────────────────────────────────────────────────
uv run python create_submission.py \
    --pred_dir "$PRED_DIR" \
    --output   "$OUTPUT"

echo "========================================"
echo "Job $SLURM_JOB_ID done. Submission: $OUTPUT"
