#!/bin/bash
#SBATCH --job-name=cil-depth
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1

module load cuda/12.6.0

# ── uv env on scratch ────────────────────────────────────────────────────────
SCRATCH_ENV="/work/scratch/cdeubel/CIL/.venv"
export UV_PROJECT_ENVIRONMENT="$SCRATCH_ENV"

# install uv if not on PATH
if ! command -v uv &>/dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
fi

PROJECT_DIR="$HOME/CIL"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

uv sync --frozen

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR="/cluster/courses/cil/monocular-depth-estimation"
BASE_CONFIG="${1:-configs/config.yaml}"

echo "Job:    $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: $BASE_CONFIG"
echo "Env:    $SCRATCH_ENV"

# patch data_root to cluster path without touching the committed config
PATCHED_CONFIG="logs/config_${SLURM_JOB_ID}.yaml"
uv run python - <<EOF
import yaml
with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["data_root"] = "$DATA_DIR"
with open("$PATCHED_CONFIG", "w") as f:
    yaml.dump(cfg, f)
EOF

uv run python src/train.py --config "$PATCHED_CONFIG"
