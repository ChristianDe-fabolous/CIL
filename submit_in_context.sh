#!/bin/bash
# ── ETH ISG Student Cluster — CIL In-Context Depth Estimation ────────────────
#
# Usage:
#   sbatch submit_in_context.sh
#
# Override defaults via env vars at sbatch time:
#   STRATEGY=patch_sim  sbatch submit_in_context.sh
#   EPOCHS=10           sbatch submit_in_context.sh
#
# Smoke test (1 epoch, small batch):
#   SMOKE=1 sbatch submit_in_context.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=CIL-incontext
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --account=cil_jobs
#SBATCH --gpus=3090:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=cdeubel@ethz.ch
#SBATCH --mail-type=END,FAIL

# ── GPU reference ─────────────────────────────────────────────────────────────
# in-context K=4 (bs=4, img=560)  → 3090  (~16 GB)  --gpus=3090:1
# in-context K=8 (bs=2, img=560)  → gb10  (~32 GB)  --gpus=gb10:1
# ─────────────────────────────────────────────────────────────────────────────

# ── Defaults (override via env vars at sbatch time) ───────────────────────────
STRATEGY="${STRATEGY:-random}"   # random | patch_sim | gt_depth_sim | learned
EPOCHS="${EPOCHS:-}"             # leave empty to use config default

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
echo "Strategy : $STRATEGY"
echo "Env      : $SCRATCH_ENV"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Patch data_root without touching committed config ─────────────────────────
PATCHED_CONFIG="logs/config_${SLURM_JOB_ID}.yaml"
uv run python - <<EOF
import yaml
with open("configs/config_in_context.yaml") as f:
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

# ── Run ───────────────────────────────────────────────────────────────────────
CMD="uv run python src/train_in_context.py --config $PATCHED_CONFIG --strategy $STRATEGY"

echo "CMD: $CMD"
eval $CMD

echo "========================================"
echo "Job $SLURM_JOB_ID done."
