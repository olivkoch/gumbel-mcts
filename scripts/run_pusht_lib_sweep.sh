#!/bin/bash
#SBATCH --job-name=pusht-lib
#SBATCH --partition=h100
#SBATCH --qos=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/vast/home/olivier.koch/workspace/gumbel-mcts/logs/pusht-lib-%j.out
#SBATCH --error=/mnt/vast/home/olivier.koch/workspace/gumbel-mcts/logs/pusht-lib-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /mnt/vast/home/olivier.koch/workspace/gumbel-mcts

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'not visible yet')"
echo "Start: $(date)"
echo ""

uv run --python 3.12 \
    --with gymnasium --with "gym-pusht" --with "pymunk<7" \
    --with "lerobot[diffusion]" --with huggingface_hub --with safetensors \
    python demo/pusht_lib_sweep.py "$@"

echo ""
echo "Done: $(date)"
