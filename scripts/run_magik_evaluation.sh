#!/bin/bash
# ==============================================================================
# Script: run_magik_evaluation.sh
# Description: Evaluates MAGIK (M2-VAE imagination baseline) across all seeds
#              on MiniWorld Case 3 (target3) and consolidates data into:
#                - Results/baselines/magik.md
#                - Results/baselines/magik_cache.json
#                - Results/diagnostics/MAGIK_MiniWorld_target3_seed_*.json
# ==============================================================================

set -uo pipefail

# ------------------------------------------------------------------------------
# 1. Environment & Python Setup
# ------------------------------------------------------------------------------
if [ -f "/home/s222147455/.bashrc" ]; then
    source /home/s222147455/.bashrc 2>/dev/null || true
fi

if command -v module &> /dev/null; then
    module load Anaconda3 2>/dev/null || true
fi

if command -v conda &> /dev/null; then
    conda activate lg_magik 2>/dev/null || eval "$(conda shell.bash hook)" && conda activate lg_magik 2>/dev/null || true
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

# ------------------------------------------------------------------------------
# 2. Default Parameters
# ------------------------------------------------------------------------------
SEEDS=(42 123 456 789 1024)
NUM_EPISODES=10
ENV_NAME="MiniWorld"
TASK_MODE="target3"
AGENT_NAME="PPO"
OUTPUT_DIR="Results/baselines"
LOG_DIR="Results/logs/baselines"
AUTO_RESUME=true
CONTINUE_ON_ERROR=true
EXTRA_ARGS=()

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "Results/diagnostics"

# ------------------------------------------------------------------------------
# 3. Parse Command-Line Options
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)
            read -r -a SEEDS <<< "$2"
            shift 2
            ;;
        --seed)
            SEEDS=("$2")
            shift 2
            ;;
        --episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume)
            AUTO_RESUME=true
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
        --help|-h)
            echo "Usage: $0 [options] [-- <extra hydra arguments>]"
            echo ""
            echo "Options:"
            echo "  --seeds \"42 123 456 789 1024\"  Space-separated list of seeds (default: 42 123 456 789 1024)"
            echo "  --seed <int>                   Evaluate a single seed"
            echo "  --episodes <int>               Number of test episodes per seed (default: 10)"
            echo "  --output-dir <path>            Directory for output report (default: Results/baselines)"
            echo "  --resume                       Skip already completed seeds (default: true)"
            echo "  --no-resume, --force           Re-evaluate and overwrite existing cached runs"
            echo "  --fail-fast                    Exit immediately on first failure"
            echo "  --help, -h                     Show this help message"
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

MD_REPORT="${OUTPUT_DIR}/magik.md"
CACHE_FILE="${OUTPUT_DIR}/magik_cache.json"

# Helper function to check if a run has already completed in the cache
is_run_completed() {
    local cache_path="$1"
    local seed_val="$2"
    local min_eps="$3"

    if [ ! -f "$cache_path" ]; then
        return 1
    fi

    "$PYTHON_BIN" -c "
import sys, json
try:
    with open('$cache_path', 'r') as f:
        records = json.load(f)
    for r in records:
        if (r.get('env_name') == '$ENV_NAME' and
            r.get('task_mode') == '$TASK_MODE' and
            r.get('seed') == int('$seed_val') and
            'MAGIK' in r.get('agent_name', '')):
            ep_keys = [k for k in r if k.startswith('episode_')]
            if len(ep_keys) >= int('$min_eps') or r.get('total_episodes', 0) >= int('$min_eps'):
                sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
" >/dev/null 2>&1
    return $?
}

echo "================================================================================"
echo "          MAGIK (M2-VAE) BASELINE EVALUATION - MINIWORLD CASE 3"
echo "================================================================================"
echo "Environment         : $ENV_NAME"
echo "Task Mode           : $TASK_MODE (Case 3: pick green ball, avoid blue box)"
echo "Source Agent        : $AGENT_NAME"
echo "Seeds to Evaluate   : ${SEEDS[*]}"
echo "Episodes per Seed   : $NUM_EPISODES"
echo "Consolidated Report : $MD_REPORT"
echo "Cache File          : $CACHE_FILE"
echo "Auto-Resume         : $AUTO_RESUME"
echo "Python Executable   : $PYTHON_BIN"
echo "================================================================================"

TOTAL_SEEDS=${#SEEDS[@]}
CURRENT=0
SUCCESS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

for SEED in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    VAE_CKPT="model_weights/${ENV_NAME}/MAGIK/m2_vae/seed_${SEED}/model.pt"
    AGENT_CKPT="model_weights/${ENV_NAME}/${AGENT_NAME}/seed_${SEED}/${AGENT_NAME}.zip"
    LOG_FILE="${LOG_DIR}/magik_${ENV_NAME}_${TASK_MODE}_seed_${SEED}.log"

    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "[Progress: ${CURRENT}/${TOTAL_SEEDS}] Evaluating Seed: ${SEED}"
    echo "  M2-VAE Checkpoint : ${VAE_CKPT}"
    echo "  PPO Agent Model   : ${AGENT_CKPT}"
    echo "  Log File          : ${LOG_FILE}"
    echo "--------------------------------------------------------------------------------"

    # Check VAE weights exist
    if [ ! -f "$VAE_CKPT" ]; then
        echo "[ERROR] MAGIK M2-VAE model not found at ${VAE_CKPT}. Skipping seed ${SEED}."
        FAIL_COUNT=$((FAIL_COUNT + 1))
        if [ "$CONTINUE_ON_ERROR" = false ]; then exit 1; fi
        continue
    fi

    # Check PPO agent weights exist
    if [ ! -f "$AGENT_CKPT" ]; then
        echo "[ERROR] Pre-trained source PPO agent not found at ${AGENT_CKPT}. Skipping seed ${SEED}."
        FAIL_COUNT=$((FAIL_COUNT + 1))
        if [ "$CONTINUE_ON_ERROR" = false ]; then exit 1; fi
        continue
    fi

    # Check if seed has already completed
    if [ "$AUTO_RESUME" = true ]; then
        if is_run_completed "$CACHE_FILE" "$SEED" "$NUM_EPISODES"; then
            echo "[INFO] Seed ${SEED} already completed in ${CACHE_FILE}. Skipping."
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
    fi

    # Execute evaluation
    CMD=(
        "$PYTHON_BIN" -u baselines/MAGIK/MiniWorld/test_imagination_net.py
        "General.env=${ENV_NAME}"
        "General.task_mode=${TASK_MODE}"
        "General.seed=${SEED}"
        "Test.seed=${SEED}"
        "Test.is_agent=false"
        "Test.num_episode=${NUM_EPISODES}"
        "Test.agent_name=${AGENT_NAME}"
        "Test.agent_checkpoint=${AGENT_CKPT}"
        "Test.vae_checkpoint=${VAE_CKPT}"
        "Test.performance_md_file=${MD_REPORT}"
        "Test.cache_file=${CACHE_FILE}"
        "Test.run_Swap_viz=false"
    )

    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo "[CMD] ${CMD[*]}"
    
    # Run and log output simultaneously
    set +e
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[SUCCESS] Seed ${SEED} finished successfully."
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "[ERROR] Seed ${SEED} failed with exit code ${EXIT_CODE}. Check log: ${LOG_FILE}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        if [ "$CONTINUE_ON_ERROR" = false ]; then
            echo "[ABORT] Exiting on error (--fail-fast enabled)."
            exit 1
        fi
    fi
done

echo ""
echo "================================================================================"
echo "                   MAGIK EVALUATION SUMMARY"
echo "================================================================================"
echo "Total Seeds Processed : ${TOTAL_SEEDS}"
echo "Successful Runs       : ${SUCCESS_COUNT}"
echo "Skipped (Completed)   : ${SKIP_COUNT}"
echo "Failed Runs           : ${FAIL_COUNT}"
echo "Consolidated Report   : ${MD_REPORT}"
echo "Consolidated Cache    : ${CACHE_FILE}"
echo "================================================================================"

if [ -f "$MD_REPORT" ]; then
    echo ""
    echo "--- Consolidated Performance Summary (${MD_REPORT}) ---"
    cat "$MD_REPORT"
fi
