#!/bin/bash
# ==============================================================================
# Script: eval_multimodal_baseline.sh
# Description: Evaluates multimodal baseline models on MiniWorld Case 3 (target3)
#              Baselines supported:
#                1. finetuned_20k  - Evaluates at 20,000 fine-tuning steps
#                2. finetuned_50k  - Evaluates at 50,000 fine-tuning steps
#                3. direct         - Zero-shot direct transfer (source model)
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

SEEDS=(42 123 456 789 1024)
NUM_EPISODES=10
BASELINE_FILTER="all"   # "all", "finetuned_20k", "finetuned_50k", "direct"
OUTPUT_DIR="Results/baselines"
LOG_DIR="Results/logs/baselines"
AUTO_RESUME=true
CONTINUE_ON_ERROR=true

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline)
            BASELINE_FILTER="$2"
            shift 2
            ;;
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
echo "          MINIWORLD MULTIMODAL BASELINE EVALUATION (CASE 3: target3)"
echo "================================================================================"
echo "Python Executable : $PYTHON_BIN"
echo "Baseline Filter   : $BASELINE_FILTER"
echo "Seeds             : ${SEEDS[*]}"
echo "Episodes / Seed   : $NUM_EPISODES"
echo "Output Directory  : $OUTPUT_DIR"
echo "Auto Resume       : $AUTO_RESUME"
echo "================================================================================"

is_already_evaluated() {
    local cache_file="$1"
    local seed_val="$2"
    local agent_tag="$3"

    if [ ! -f "$cache_file" ]; then
        return 1
    fi

    "$PYTHON_BIN" -c "
import sys, json
try:
    with open('$cache_file', 'r') as f:
        records = json.load(f)
    for r in records:
        if (r.get('env_name') == 'MiniWorld' and
            r.get('task_mode') == 'target3' and
            r.get('seed') == int('$seed_val') and
            r.get('agent_name') == '$agent_tag'):
            ep_keys = [k for k in r if k.startswith('episode_')]
            if len(ep_keys) >= int('$NUM_EPISODES') or r.get('total_episodes', 0) >= int('$NUM_EPISODES'):
                sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
" >/dev/null 2>&1
    return $?
}

run_eval_case() {
    local b_tag="$1"
    local agent_tag="$2"
    local seed_val="$3"
    local model_path="$4"
    local md_out="$5"

    local cache_file="${md_out%.*}_cache.json"
    local log_file="${LOG_DIR}/${b_tag}_MiniWorld_target3_seed_${seed_val}.log"

    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "[Baseline: ${b_tag}] MiniWorld target3 | Seed: ${seed_val}"
    echo "Agent Tag : ${agent_tag}"
    echo "Model Path: ${model_path}"
    echo "MD Output : ${md_out}"
    echo "--------------------------------------------------------------------------------"

    if [ ! -f "$model_path" ]; then
        echo "[ERROR] Model checkpoint not found at: ${model_path}. Skipping."
        if [ "$CONTINUE_ON_ERROR" = false ]; then exit 1; fi
        return 1
    fi

    if [ "$AUTO_RESUME" = true ]; then
        if is_already_evaluated "$cache_file" "$seed_val" "$agent_tag"; then
            echo "[INFO] Already evaluated in ${cache_file}. Skipping."
            return 0
        fi
    fi

    START_T=$(date +%s)
    if "$PYTHON_BIN" -u test_imagination.py \
        env=MiniWorld \
        env.task_mode=target3 \
        env.observation_mode=multimodal \
        agent_name="${agent_tag}" \
        dqn_model_dir="${model_path}" \
        mode=source \
        seed="${seed_val}" \
        num_episode="${NUM_EPISODES}" \
        performance_md_file="${md_out}" 2>&1 | tee "$log_file"; then
        END_T=$(date +%s)
        echo "[SUCCESS] Evaluation finished in $((END_T - START_T))s. Saved to ${md_out}"
    else
        EXIT_CODE=$?
        END_T=$(date +%s)
        echo "[ERROR] Evaluation failed with code ${EXIT_CODE} after $((END_T - START_T))s."
        if [ "$CONTINUE_ON_ERROR" = false ]; then exit $EXIT_CODE; fi
        return $EXIT_CODE
    fi
}

# ------------------------------------------------------------------------------
# 1. Direct Transfer (Pretrained Source Agent)
# ------------------------------------------------------------------------------
if [ "$BASELINE_FILTER" == "all" ] || [ "$BASELINE_FILTER" == "direct" ]; then
    echo ""
    echo ">>> Running Baseline: Direct Zero-Shot Multimodal"
    MD_FILE="${OUTPUT_DIR}/direct_multimodal.md"
    for S_VAL in "${SEEDS[@]}"; do
        CKPT="model_weights/MiniWorld/PPO_multimodal/seed_${S_VAL}/PPO_multimodal.zip"
        run_eval_case "direct_multimodal" "Direct_PPO_multimodal" "$S_VAL" "$CKPT" "$MD_FILE"
    done
fi

# ------------------------------------------------------------------------------
# 2. Fine-Tuned 20K Steps
# ------------------------------------------------------------------------------
if [ "$BASELINE_FILTER" == "all" ] || [ "$BASELINE_FILTER" == "finetuned_20k" ]; then
    echo ""
    echo ">>> Running Baseline: Fine-Tuned 20K Steps Multimodal"
    MD_FILE="${OUTPUT_DIR}/finetuned_20k_multimodal.md"
    for S_VAL in "${SEEDS[@]}"; do
        CKPT="model_weights/MiniWorld/finetuned/target3/PPO_multimodal/seed_${S_VAL}/PPO_multimodal_20000_steps.zip"
        run_eval_case "finetuned_20k_multimodal" "Finetuned20k_PPO_multimodal" "$S_VAL" "$CKPT" "$MD_FILE"
    done
fi

# ------------------------------------------------------------------------------
# 3. Fine-Tuned 50K Steps
# ------------------------------------------------------------------------------
if [ "$BASELINE_FILTER" == "all" ] || [ "$BASELINE_FILTER" == "finetuned_50k" ]; then
    echo ""
    echo ">>> Running Baseline: Fine-Tuned 50K Steps Multimodal"
    MD_FILE="${OUTPUT_DIR}/finetuned_50k_multimodal.md"
    for S_VAL in "${SEEDS[@]}"; do
        CKPT="model_weights/MiniWorld/finetuned/target3/PPO_multimodal/seed_${S_VAL}/PPO_multimodal_50000_steps.zip"
        if [ ! -f "$CKPT" ]; then
            CKPT="model_weights/MiniWorld/finetuned/target3/PPO_multimodal/seed_${S_VAL}/PPO_multimodal.zip"
        fi
        run_eval_case "finetuned_50k_multimodal" "Finetuned50k_PPO_multimodal" "$S_VAL" "$CKPT" "$MD_FILE"
    done
fi

echo ""
echo "================================================================================"
echo "          MINIWORLD MULTIMODAL BASELINE EVALUATION COMPLETE"
echo "================================================================================"
