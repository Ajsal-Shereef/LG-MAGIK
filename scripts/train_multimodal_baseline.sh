#!/bin/bash
# ==============================================================================
# Script: train_multimodal_baseline.sh
# Description: Trains multimodal PPO baseline for MiniWorld (Case 3):
#              Modality: Image (80x80x3) + Vector (4D presence & distance)
#              Vector: [<obj1_pres>, <obj2_pres>, <obj1_dist>, <obj2_dist>]
#
#              Stage 1: Pre-train source agent on MiniWorld (task_mode=source)
#                       for 200,000 steps.
#              Stage 2: Fine-tune pre-trained agent on MiniWorld Case 3 (target3)
#                       for 50,000 steps (saving checkpoints at 10k, 20k, 30k, 40k, 50k).
# ==============================================================================

set -uo pipefail

if [ -f "/home/s222147455/.bashrc" ]; then
    source /home/s222147455/.bashrc 2>/dev/null || true
fi

if command -v conda &> /dev/null; then
    source activate 2>/dev/null || eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate lg_magik 2>/dev/null || true
fi

PYTHON_BIN="python"
if [ -x "/home/s222147455/.conda/lg_magik/bin/python" ]; then
    PYTHON_BIN="/home/s222147455/.conda/lg_magik/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi

PROJECT_ROOT="/weka/s222147455/LG-MAGIK"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# Defaults
SEEDS=(42 123 456 789 1024)
STAGE="both"               # "source", "finetune", "both"
SOURCE_STEPS=200000
FINETUNE_STEPS=50000
USE_WANDB="true"
AUTO_RESUME=true
CONTINUE_ON_ERROR=true
LOG_DIR="Results/logs/multimodal"
mkdir -p "$LOG_DIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            SEEDS=("$2")
            shift 2
            ;;
        --seeds)
            read -r -a SEEDS <<< "$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --source-steps)
            SOURCE_STEPS="$2"
            shift 2
            ;;
        --finetune-steps)
            FINETUNE_STEPS="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB="true"
            shift
            ;;
        --no-wandb)
            USE_WANDB="false"
            shift
            ;;
        --no-resume|--force)
            AUTO_RESUME=false
            shift
            ;;
        --fail-fast)
            CONTINUE_ON_ERROR=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "          MINIWORLD MULTIMODAL BASELINE TRAINING (SOURCE + FINETUNING)"
echo "================================================================================"
echo "Python Executable : $PYTHON_BIN"
echo "Stage             : $STAGE"
echo "Seeds             : ${SEEDS[*]}"
echo "Source Steps      : $SOURCE_STEPS"
echo "Fine-tuning Steps : $FINETUNE_STEPS"
echo "Wandb             : $USE_WANDB"
echo "Auto Resume       : $AUTO_RESUME"
echo "Log Directory     : $LOG_DIR"
echo "================================================================================"

# ------------------------------------------------------------------------------
# STAGE 1: SOURCE PRE-TRAINING
# ------------------------------------------------------------------------------
if [ "$STAGE" == "source" ] || [ "$STAGE" == "both" ]; then
    echo ""
    echo "================================================================================"
    echo ">>> STAGE 1: Training Source Multimodal PPO Agents (${SOURCE_STEPS} steps)"
    echo "================================================================================"

    for SEED in "${SEEDS[@]}"; do
        SOURCE_DIR="model_weights/MiniWorld/PPO_multimodal/seed_${SEED}"
        SOURCE_MODEL="${SOURCE_DIR}/PPO_multimodal.zip"
        LOG_FILE="${LOG_DIR}/train_source_seed_${SEED}.log"

        if [ "$AUTO_RESUME" = true ] && [ -f "$SOURCE_MODEL" ]; then
            echo "[INFO] Source model already exists at: ${SOURCE_MODEL}. Skipping."
            continue
        fi

        echo "[Source] Training Seed ${SEED} -> ${SOURCE_MODEL}"
        START_T=$(date +%s)
        if "$PYTHON_BIN" train_agent.py \
            env=MiniWorld \
            env.task_mode=source \
            env.observation_mode=multimodal \
            agent_name=PPO_multimodal \
            seed="${SEED}" \
            fine_tune=false \
            env.total_timestep="${SOURCE_STEPS}" \
            save_data=false \
            use_wandb="${USE_WANDB}" 2>&1 | tee "$LOG_FILE"; then
            END_T=$(date +%s)
            echo "[SUCCESS] Source seed ${SEED} completed in $((END_T - START_T))s."
        else
            EXIT_CODE=$?
            echo "[ERROR] Source training failed for seed ${SEED} with code ${EXIT_CODE}."
            if [ "$CONTINUE_ON_ERROR" = false ]; then exit $EXIT_CODE; fi
        fi
    done
fi

# ------------------------------------------------------------------------------
# STAGE 2: TARGET3 FINE-TUNING
# ------------------------------------------------------------------------------
if [ "$STAGE" == "finetune" ] || [ "$STAGE" == "both" ]; then
    echo ""
    echo "================================================================================"
    echo ">>> STAGE 2: Fine-Tuning Multimodal PPO on Case 3 (target3) (${FINETUNE_STEPS} steps)"
    echo "================================================================================"

    for SEED in "${SEEDS[@]}"; do
        SOURCE_MODEL="model_weights/MiniWorld/PPO_multimodal/seed_${SEED}/PPO_multimodal.zip"
        FT_DIR="model_weights/MiniWorld/finetuned/target3/PPO_multimodal/seed_${SEED}"
        FT_FINAL="${FT_DIR}/PPO_multimodal.zip"
        FT_20K="${FT_DIR}/PPO_multimodal_20000_steps.zip"
        FT_50K="${FT_DIR}/PPO_multimodal_50000_steps.zip"
        LOG_FILE="${LOG_DIR}/finetune_target3_seed_${SEED}.log"

        if [ ! -f "$SOURCE_MODEL" ]; then
            echo "[ERROR] Pre-trained source model missing at: ${SOURCE_MODEL}. Cannot fine-tune."
            if [ "$CONTINUE_ON_ERROR" = false ]; then exit 1; fi
            continue
        fi

        if [ "$AUTO_RESUME" = true ] && ([ -f "$FT_50K" ] || [ -f "$FT_FINAL" ]); then
            echo "[INFO] Fine-tuned model already exists at: ${FT_DIR}. Skipping."
            continue
        fi

        echo "[Fine-tune] Training Seed ${SEED} on target3 -> ${FT_DIR}"
        START_T=$(date +%s)
        if "$PYTHON_BIN" train_agent.py \
            env=MiniWorld \
            env.task_mode=target3 \
            env.observation_mode=multimodal \
            agent_name=PPO_multimodal \
            seed="${SEED}" \
            fine_tune=true \
            fine_tune_checkpoint="${SOURCE_MODEL}" \
            env.total_timestep="${FINETUNE_STEPS}" \
            save_data=false \
            use_wandb="${USE_WANDB}" 2>&1 | tee "$LOG_FILE"; then
            END_T=$(date +%s)
            echo "[SUCCESS] Fine-tuning seed ${SEED} completed in $((END_T - START_T))s."
        else
            EXIT_CODE=$?
            echo "[ERROR] Fine-tuning failed for seed ${SEED} with code ${EXIT_CODE}."
            if [ "$CONTINUE_ON_ERROR" = false ]; then exit $EXIT_CODE; fi
        fi
    done
fi

echo ""
echo "================================================================================"
echo "          MINIWORLD MULTIMODAL BASELINE TRAINING COMPLETE"
echo "================================================================================"
