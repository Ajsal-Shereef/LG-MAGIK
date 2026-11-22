#!/bin/bash
#SBATCH --job-name=eval_magik_miniworld
#SBATCH --output=slurm-%x-%A.out
#SBATCH --error=slurm-%x-%A.err
#SBATCH --partition=gpu-large,gpu
#SBATCH --qos=batch-short
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --cpus-per-gpu=4
#SBATCH --time=12:00:00

# ==============================================================================
# Script: run_magik_evaluation_sbatch.sh
# Usage:
#   sbatch scripts/run_magik_evaluation_sbatch.sh [options]
#
# Examples:
#   sbatch scripts/run_magik_evaluation_sbatch.sh
#   sbatch scripts/run_magik_evaluation_sbatch.sh --seeds "42 123 456"
#   sbatch scripts/run_magik_evaluation_sbatch.sh --episodes 10 --force
# ==============================================================================

module purge
module load Anaconda3 2>/dev/null || true
cd /home/s222147455
if [ -f "/home/s222147455/.bashrc" ]; then
    source /home/s222147455/.bashrc 2>/dev/null || true
fi
conda activate lg_magik 2>/dev/null || true

cd "/weka/s222147455/LG-MAGIK"

echo "================================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting MAGIK evaluation SLURM job"
echo "Job ID:       ${SLURM_JOB_ID:-manual}"
echo "Node:         ${SLURM_NODELIST:-localhost}"
echo "Partition:    ${SLURM_JOB_PARTITION:-unknown}"
echo "Arguments:    $*"
echo "================================================================================"

./scripts/run_magik_evaluation.sh "$@"

echo "================================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] MAGIK evaluation job finished (Job ID: ${SLURM_JOB_ID:-manual})"
echo "================================================================================"
