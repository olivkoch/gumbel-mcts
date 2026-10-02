#!/bin/bash
#SBATCH --job-name=pusht-gifs
#SBATCH --partition=h100
#SBATCH --qos=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/vast/home/olivier.koch/workspace/gumbel-mcts/logs/pusht-gifs-%j.out
#SBATCH --error=/mnt/vast/home/olivier.koch/workspace/gumbel-mcts/logs/pusht-gifs-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /mnt/vast/home/olivier.koch/workspace/gumbel-mcts

UV="uv run --python 3.12
    --with gymnasium --with gym-pusht --with pymunk<7
    --with lerobot[diffusion] --with huggingface_hub --with safetensors"

for seed in 2782 316 42; do
    echo "=== seed=$seed ==="
    $UV python demo/pusht.py \
        --budget 128 --rollout-depth 3 --seed $seed --n-macros 10
    mv demo/pusht_comparison.gif "demo/pusht_rollout_seed${seed}.gif"
done

echo "Done: $(date)"
