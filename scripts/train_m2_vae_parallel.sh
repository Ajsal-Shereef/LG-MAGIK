#!/bin/bash
#SBATCH --job-name=train_vae_array
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=gpu-large,gpu
#SBATCH --qos=batch-short
#SBATCH --array=0-4
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=60:00:00 

module purge
module load Anaconda3
cd /home/s222147455
source /home/s222147455/.bashrc
source activate
conda activate lg_magik

cd "/weka/s222147455/LG-MAGIK/"

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-8}}
export MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-8}}

# Array of 5 seeds mapped to SLURM_ARRAY_TASK_ID (0 to 4)
SEEDS=(42 123 456 789 1024)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================================="
echo "Task ID: ${SLURM_ARRAY_TASK_ID} | Selected Seed: ${SEED}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting MAGIK M2-VAE training with seed=${SEED}"
echo "=========================================================="

python baselines/MAGIK/MiniWorld/m2_vae.py \
    General.seed=${SEED} \
    General.env=MiniWorld \
    General.data_sub_dir=MAGIK \
    "$@"
