#!/bin/bash
#SBATCH --job-name=final-vid
#SBATCH --partition=h100
#SBATCH --qos=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/vast/home/olivier.koch/workspace/gumbel-mcts/logs/final-vid-%j.out
#SBATCH --error=/mnt/vast/home/olivier.koch/workspace/gumbel-mcts/logs/final-vid-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /mnt/vast/home/olivier.koch/workspace/gumbel-mcts

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Start: $(date)"

uv run --python 3.12 \
    --with gymnasium --with "gym-pusht" --with "pymunk<7" \
    --with "lerobot[diffusion]" --with huggingface_hub --with safetensors \
    python demo/generate_wallgap_video.py

echo "Done: $(date)"
