#!/bin/bash
# ── ETH ISG Student Cluster — CIL Monocular Depth Estimation ─────────────────
#
# Usage:
#   sbatch submit.sh
#
# Override defaults via env vars at sbatch time:
#   SCRIPT=train_in_context  sbatch submit.sh
#   STRATEGY=patch_sim       sbatch submit.sh
#   DECODER=conv             sbatch submit.sh
#   EPOCHS=10                sbatch submit.sh
#
# Smoke test (1 epoch, small batch):
#   SMOKE=1 sbatch submit.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=CIL-detph
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --account=cil_jobs
#SBATCH --gpus=2080ti:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=cdeubel@ethz.ch
#SBATCH --mail-type=END,FAIL

# ── GPU reference ─────────────────────────────────────────────────────────────
# baseline  (bs=8, img=560)  → 2080ti  (~11 GB)  --gpus=2080ti:1
# in-context(bs=4, img=560)  → 3090    (~16 GB)  --gpus=3090:1
# ─────────────────────────────────────────────────────────────────────────────

# ── Defaults (override via env vars at sbatch time) ───────────────────────────
SCRIPT="${SCRIPT:-train}"                     # train | train_in_context
STRATEGY="${STRATEGY:-random}"                # random | patch_sim | gt_depth_sim | learned
DECODER="${DECODER:-transformer}"             # transformer | conv  (baseline only)
EPOCHS="${EPOCHS:-}"                          # leave empty to use config default
CONFIG="${CONFIG:-}"                          # override config file (e.g. configs/config_base.yaml)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO="/work/scratch/cdeubel/CIL"
DATA_DIR="/cluster/courses/cil/monocular-depth-estimation"
SCRATCH_ENV="/work/scratch/cdeubel/CIL/.venv"

# ── Environment ───────────────────────────────────────────────────────────────
module load cuda/12.6.0
export UV_PROJECT_ENVIRONMENT="$SCRATCH_ENV"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ! command -v uv &>/dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$REPO"
mkdir -p logs checkpoints

uv sync --frozen

# ── Job info ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Script   : src/${SCRIPT}.py"
echo "Strategy : $STRATEGY"
echo "Env      : $SCRATCH_ENV"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Patch data_root without touching committed config ─────────────────────────
if [ -n "$CONFIG" ]; then
    BASE_CONFIG="$CONFIG"
elif [ "$SCRIPT" = "train_in_context" ]; then
    BASE_CONFIG="configs/config_in_context.yaml"
else
    BASE_CONFIG="configs/config.yaml"
fi

PATCHED_CONFIG="logs/config_${SLURM_JOB_ID}.yaml"
uv run python - <<EOF
import yaml
with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["data_root"] = "$DATA_DIR"
if "$EPOCHS":
    cfg["training"]["epochs"] = int("$EPOCHS")
if "$SMOKE" == "1":
    cfg["training"]["epochs"] = 1
    cfg["training"]["batch_size"] = 2
with open("$PATCHED_CONFIG", "w") as f:
    yaml.dump(cfg, f)
EOF

# ── Build command ─────────────────────────────────────────────────────────────
CMD="uv run python src/${SCRIPT}.py --config $PATCHED_CONFIG"

[ "$SCRIPT" = "train" ]            && CMD="$CMD --decoder $DECODER"
[ "$SCRIPT" = "train_in_context" ] && CMD="$CMD --strategy $STRATEGY"

echo "CMD: $CMD"
eval $CMD

# ── Post-training: predict + submit (baseline train only) ────────────────────
if [ "$SCRIPT" = "train" ]; then
    # find the checkpoint written by this job
    CKPT=$(find checkpoints -name "best.pth" -newer "$PATCHED_CONFIG" | head -1)
    if [ -z "$CKPT" ]; then
        echo "WARNING: no checkpoint found, skipping predict+submit"
    else
        PRED_DIR="predictions/${SLURM_JOB_ID}"
        SUB_CSV="submissions/${SLURM_JOB_ID}/submission.csv"
        mkdir -p "$(dirname "$SUB_CSV")"

        echo "--- Predicting with $CKPT ---"
        uv run python src/predict.py \
            --checkpoint "$CKPT" \
            --test_dir   "$DATA_DIR/test" \
            --output_dir "$PRED_DIR"

        echo "--- Building submission CSV ---"
        uv run python create_submission.py \
            --pred_dir "$PRED_DIR" \
            --output   "$SUB_CSV"

        echo "--- Uploading to Kaggle ---"
        uv run kaggle competitions submit \
            -c ethz-cil-monocular-depth-estimation-2026 \
            -f "$SUB_CSV" \
            -m "job ${SLURM_JOB_ID} / ${DECODER} decoder / $(date +%F)"
    fi
fi

echo "========================================"
echo "Job $SLURM_JOB_ID done."
