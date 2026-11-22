#!/bin/bash
#SBATCH --job-name=train_m2_vae_single
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=gpu-large,gpu
#SBATCH --qos=batch-short
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=60:00:00

module purge
module load Anaconda3
cd /home/s222147455
source /home/s222147455/.bashrc
conda activate lg_magik

cd "/weka/s222147455/LG-MAGIK/"

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-8}}
export MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-8}}

SEED=${1:-42}
shift 1 2>/dev/null || true

echo "=========================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting MAGIK M2-VAE single seed=${SEED}"
echo "=========================================================="

python baselines/MAGIK/MiniWorld/m2_vae.py \
    General.seed=${SEED} \
    General.env=MiniWorld \
    General.data_sub_dir=MAGIK \
    "$@"
