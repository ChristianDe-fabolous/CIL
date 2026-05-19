#!/bin/bash
#SBATCH --job-name=cil-depth
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1

# ── conda ──────────────────────────────────────────────────────────────────
__conda_setup="$('/cluster/courses/cil/envs/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cluster/courses/cil/envs/etc/profile.d/conda.sh" ]; then
        . "/cluster/courses/cil/envs/etc/profile.d/conda.sh"
    else
        export PATH="/cluster/courses/cil/envs/bin:$PATH"
    fi
fi
unset __conda_setup

module load cuda/12.6.0
conda activate /cluster/courses/cil/envs/envs/monocular-depth-estimation

# ── paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/CIL"          # adjust if repo lives elsewhere on cluster
DATA_DIR="/cluster/courses/cil/monocular-depth-estimation"
BASE_CONFIG="${1:-configs/config.yaml}"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints

echo "Job:    $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: $BASE_CONFIG"
echo "Data:   $DATA_DIR"

# patch data_root to cluster path without touching the committed config
PATCHED_CONFIG="logs/config_${SLURM_JOB_ID}.yaml"
python - <<EOF
import yaml, sys
with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["data_root"] = "$DATA_DIR"
with open("$PATCHED_CONFIG", "w") as f:
    yaml.dump(cfg, f)
EOF

python src/train.py --config "$PATCHED_CONFIG"
