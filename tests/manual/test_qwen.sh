#!/bin/bash
# Qwen3 training tests - GPU pool scheduler, parallel execution, per-task log files
#
# Auto-detects available GPUs and allocates on demand.
# Queues tasks when GPUs are insufficient, launches as GPUs are freed.
# CUDA_VISIBLE_DEVICES ensures full GPU isolation between tasks.
#
# Usage:
#   bash tests/test_qwen.sh              # default 3 steps, auto-detect GPUs
#   bash tests/test_qwen.sh 10           # custom step count
#   bash tests/test_qwen.sh 3 logs/run1  # custom steps + log directory
#   NUM_GPUS=4 bash tests/test_qwen.sh   # manually specify total GPU count
#
#   GPU pool indices 0..N-1 are passed to CUDA_VISIBLE_DEVICES for each task.
#   To map pool slot 0 to physical GPU 1 (e.g. GPU 0 is busy), set:
#     GPU_POOL_OFFSET=1 NUM_GPUS=1 bash tests/manual/test_qwen.sh
#   (Exporting CUDA_VISIBLE_DEVICES before this script is overwritten at launch; use OFFSET instead.)

# set -euo pipefail

# STEPS=${1:-5}
# SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# LOG_DIR=${2:-$SCRIPT_DIR/logs/$(date +%Y%m%d_%H%M%S)}
# SCRIPT=tests/test_qwen3_train.py
STEPS=5
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="$SCRIPT_DIR/logs/$(date +%Y%m%d_%H%M%S)"
SCRIPT=tests/manual/test_qwen3_train.py
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-/data/nfs/Qwen3-0.6B}
# Added to each pool GPU id when setting CUDA_VISIBLE_DEVICES (default 0).
GPU_POOL_OFFSET=${GPU_POOL_OFFSET:-0}

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --model|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --python-bin)
            PYTHON_BIN="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash tests/manual/test_qwen.sh [--steps N] [--log-dir DIR] [--model PATH] [--python-bin PYTHON]"
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

TOTAL_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}

mkdir -p "$LOG_DIR"

echo "========================================"
echo " Qwen3 Training Tests (GPU pool)"
echo " Steps:      $STEPS"
echo " Total GPUs: $TOTAL_GPUS"
echo " Log dir:    $LOG_DIR"
echo "========================================"

# ---------------------------------------------------------------------------
# Task definitions: name|num_gpus|command
# CUDA_VISIBLE_DEVICES is assigned by the scheduler; torchrun sees GPUs 0..n-1
# ---------------------------------------------------------------------------
declare -a TASKS=(
    "cuda_single|1|$PYTHON_BIN $SCRIPT --device cuda --model $MODEL_PATH --steps $STEPS"
    "flagos_single|1|$PYTHON_BIN $SCRIPT --device flagos --model $MODEL_PATH --steps $STEPS --flagos-qwen3-attention custom"
    # Re-enable FlagGems / Triton when kernels are fixed for your GPU + driver:
    "flagos_single_gems|1|ATTENTION_PROBE=1 $PYTHON_BIN $SCRIPT --device flagos --model $MODEL_PATH --steps $STEPS --debug-bwd-hooks --attention-probe --debug-dump-dir ./debug_dump --debug-attn-q-path --debug-attn-q-path-bwd"
    "cuda_ddp_nccl|2|torchrun --nproc_per_node=2 --master_port=29500 $SCRIPT --device cuda --model $MODEL_PATH --parallel ddp --comm nccl --steps $STEPS"
    "cuda_ddp_flagcx|2|torchrun --nproc_per_node=2 --master_port=29501 $SCRIPT --device cuda --model $MODEL_PATH --parallel ddp --comm flagcx --steps $STEPS"
    "cuda_fsdp_nccl|2|torchrun --nproc_per_node=2 --master_port=29502 $SCRIPT --device cuda --model $MODEL_PATH --parallel fsdp --comm nccl --steps $STEPS"
    "cuda_fsdp_flagcx|2|torchrun --nproc_per_node=2 --master_port=29503 $SCRIPT --device cuda --model $MODEL_PATH --parallel fsdp --comm flagcx --steps $STEPS"
    "flagos_ddp_nccl|2|torchrun --nproc_per_node=2 --master_port=29504 $SCRIPT --device flagos --model $MODEL_PATH --parallel ddp --comm nccl --steps $STEPS"
    "flagos_ddp_flagcx|2|torchrun --nproc_per_node=2 --master_port=29505 $SCRIPT --device flagos --model $MODEL_PATH --parallel ddp --comm flagcx --steps $STEPS"
    "flagos_fsdp_nccl|2|torchrun --nproc_per_node=2 --master_port=29506 $SCRIPT --device flagos --model $MODEL_PATH --parallel fsdp --comm nccl --steps $STEPS"
    "flagos_fsdp_flagcx|2|torchrun --nproc_per_node=2 --master_port=29507 $SCRIPT --device flagos --model $MODEL_PATH --parallel fsdp --comm flagcx --steps $STEPS"
)

# ---------------------------------------------------------------------------
# GPU pool: track each GPU's status (0 = free, pid = occupied)
# ---------------------------------------------------------------------------
declare -a GPU_OWNER=()
for ((i = 0; i < TOTAL_GPUS; i++)); do
    GPU_OWNER[$i]=0
done

# Map of launched task pid -> name
declare -A PID_TO_NAME=()
# Result collection
declare -a DONE_NAMES=()
declare -a DONE_STATUS=()

# Try to allocate N free GPUs from the pool.
# On success: returns 0, sets ALLOC_GPUS to comma-separated GPU IDs.
# On failure: returns 1.
alloc_gpus() {
    local need=$1
    local found=()
    for ((i = 0; i < TOTAL_GPUS; i++)); do
        if [[ ${GPU_OWNER[$i]} -eq 0 ]]; then
            found+=($i)
            if [[ ${#found[@]} -eq $need ]]; then
                ALLOC_GPUS=$(IFS=,; echo "${found[*]}")
                return 0
            fi
        fi
    done
    return 1
}

# Mark GPUs as occupied by a given pid
mark_busy() {
    local gpus=$1 pid=$2
    IFS=',' read -ra ids <<< "$gpus"
    for id in "${ids[@]}"; do
        GPU_OWNER[$id]=$pid
    done
}

# Release all GPUs held by a given pid
release_gpus() {
    local pid=$1
    for ((i = 0; i < TOTAL_GPUS; i++)); do
        if [[ ${GPU_OWNER[$i]} -eq $pid ]]; then
            GPU_OWNER[$i]=0
        fi
    done
}

# Wait for any running task to finish, release its GPUs, and record result
wait_any() {
    while true; do
        for pid in "${!PID_TO_NAME[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                # Process has exited, collect exit code
                wait "$pid" 2>/dev/null
                local rc=$?
                local name="${PID_TO_NAME[$pid]}"
                release_gpus "$pid"
                unset "PID_TO_NAME[$pid]"
                DONE_NAMES+=("$name")
                DONE_STATUS+=($rc)
                if [[ $rc -eq 0 ]]; then
                    echo "[  OK  ] $name"
                else
                    echo "[ FAIL ] $name (exit=$rc)"
                fi
                return
            fi
        done
        sleep 0.5
    done
}

# ---------------------------------------------------------------------------
# Scheduling loop: launch when GPUs available, wait otherwise
# ---------------------------------------------------------------------------
for entry in "${TASKS[@]}"; do
    IFS='|' read -r name num_gpus cmd <<< "$entry"
    log_file="$LOG_DIR/${name}.log"

    # Wait until enough free GPUs are available
    while ! alloc_gpus "$num_gpus"; do
        wait_any
    done

    echo "[LAUNCH] $name  (GPUs: $ALLOC_GPUS) -> $log_file"
    _cv=""
    IFS=',' read -ra _alloc_ids <<< "$ALLOC_GPUS"
    for _id in "${_alloc_ids[@]}"; do
        _cv="${_cv:+$_cv,}$((_id + GPU_POOL_OFFSET))"
    done
    ( CUDA_VISIBLE_DEVICES="$_cv" eval "$cmd" > "$log_file" 2>&1 ) &
    local_pid=$!
    mark_busy "$ALLOC_GPUS" "$local_pid"
    PID_TO_NAME[$local_pid]="$name"
done

# Wait for remaining tasks to finish
while [[ ${#PID_TO_NAME[@]} -gt 0 ]]; do
    wait_any
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
FAIL=0
for s in "${DONE_STATUS[@]}"; do
    [[ $s -ne 0 ]] && FAIL=$((FAIL + 1))
done

echo ""
echo "========================================"
echo " Results: $(( ${#DONE_NAMES[@]} - FAIL ))/${#DONE_NAMES[@]} passed"
echo " Logs:    $LOG_DIR/"
echo "========================================"

# Print last few lines of each log as a quick summary
for entry in "${TASKS[@]}"; do
    IFS='|' read -r name _ _ <<< "$entry"
    log_file="$LOG_DIR/${name}.log"
    echo ""
    echo "--- $name (last 5 lines) ---"
    tail -5 "$log_file" 2>/dev/null || echo "(no output)"
done

exit $FAIL
