#!/bin/bash
#SBATCH --job-name=train_multimodal
#SBATCH --output=scripts/slurm-%x-%A_%a.out
#SBATCH --error=scripts/slurm-%x-%A_%a.err
#SBATCH --partition=gpu-large,gpu
#SBATCH --qos=batch-short
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00

# ==============================================================================
# Script: train_multimodal_baseline_sbatch.sh
# Usage:
#   1) Run all 5 seeds sequentially on 1 GPU:
#        sbatch scripts/train_multimodal_baseline_sbatch.sh
#   2) Run seed array (e.g. seeds 42, 123, 456, 789, 1024):
#        sbatch --array=0-4 scripts/train_multimodal_baseline_sbatch.sh
# ==============================================================================

module purge
module load Anaconda3 2>/dev/null || true
cd /home/s222147455
if [ -f "/home/s222147455/.bashrc" ]; then
    source /home/s222147455/.bashrc 2>/dev/null || true
fi
source activate 2>/dev/null || eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate lg_magik 2>/dev/null || true

cd "/weka/s222147455/LG-MAGIK"

SEEDS=(42 123 456 789 1024)

echo "================================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting multimodal training SLURM job"
echo "Job ID:       ${SLURM_JOB_ID:-manual}"
echo "Array Task:   ${SLURM_ARRAY_TASK_ID:-none}"
echo "Node:         ${SLURM_NODELIST:-localhost}"
echo "================================================================================"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
    echo "Running for seed: ${SEED} (array task ${SLURM_ARRAY_TASK_ID})"
    ./scripts/train_multimodal_baseline.sh --seed "${SEED}" --wandb "$@"
    ./scripts/eval_multimodal_baseline.sh --seed "${SEED}" "$@"
else
    ./scripts/train_multimodal_baseline.sh --wandb "$@"
    ./scripts/eval_multimodal_baseline.sh "$@"
fi
