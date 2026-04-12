#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Pipeline-Parallel fine-tuning benchmark for Qwen3-14B on 4 GPUs (no CPU offload).
#
# Usage:
#   bash finetune_qwen3-14b_4gpu_pp.sh [--gbs N] [--mbs N] [--profile]
set -eo pipefail

echo "========================================================"
echo "Qwen3-14B Fine-tuning with DeepSpeed PP on 4 GPU"
echo "========================================================"

# ── Argument parsing ──────────────────────────────────────────────────────────
GBS=256
MBS=1
GPUS_PER_NODE=4
PP_STAGES=4
PROFILE=false

usage() {
    echo "Usage: $0 [--gbs N] [--mbs N] [--profile]"
    echo "  --gbs  global train batch size (default: 256)"
    echo "  --mbs         micro-batch size per pipeline stage (default: 1)"
    echo "                micro_batches (gas) is derived as: gbs / mbs"
    echo "  --profile     enable nsys profiling (default: off)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gbs) GBS="$2"; shift 2 ;;
        --mbs)        MBS="$2";        shift 2 ;;
        --profile)    PROFILE=true;    shift   ;;
        --help|-h)    usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# ── Derive micro_batches (gas) ────────────────────────────────────────────────
# PP: train_batch() processes micro_batches micro-batches per step.
# global gbs = mbs * micro_batches
if (( MBS == 0 )) || (( GBS % MBS != 0 )); then
    echo "Error: gbs (${GBS}) must be divisible by mbs (${MBS})"
    exit 1
fi
MICRO_BATCHES=$(( GBS / MBS ))
if (( MICRO_BATCHES < 1 )); then
    echo "Error: derived micro_batches (${MICRO_BATCHES}) must be >= 1"
    exit 1
fi

# ── MODELS_PATH & auto-download ──────────────────────────────────────────────
if [ -z "${MODELS_PATH}" ]; then
    echo "Error: MODELS_PATH environment variable is not set."
    echo "  export MODELS_PATH=/path/to/your/model/cache"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_REPO="Qwen/Qwen3-14B"
MODEL_NAME="${MODELS_PATH}/${MODEL_REPO}"
MODEL_SHORT="qwen3-14b"

if [ ! -d "${MODEL_NAME}" ] || [ -z "$(ls -A "${MODEL_NAME}" 2>/dev/null)" ]; then
    echo "[INFO] ${MODEL_NAME} not found, downloading ${MODEL_REPO}..."
    mkdir -p "${MODEL_NAME}"
    if command -v hf &>/dev/null; then
        hf download "${MODEL_REPO}" --local-dir "${MODEL_NAME}"
    elif command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "${MODEL_REPO}" --local-dir "${MODEL_NAME}"
    else
        echo "Error: neither 'hf' nor 'huggingface-cli' is available."
        echo "  pip install -U huggingface_hub"
        exit 1
    fi
fi

# ── Paths & tags ─────────────────────────────────────────────────────────────
CONFIG_LABEL="pp-bs${GBS}-mbs${MBS}"
RUN_TAG="${MODEL_SHORT}_pp_${CONFIG_LABEL}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${DS_ROOT}/results/cpu_offload/${MODEL_SHORT}/pp/${CONFIG_LABEL}_${RUN_TIMESTAMP}"

mkdir -p "${OUTPUT_DIR}"

# ── Script argument parameters ────────────────────────────────────────────────
ACTIVATION_CHECKPOINTING=true
MAX_LENGTH=4096
LOG_INTERVAL=1
DATASET_NAME="tatsu-lab/alpaca"
DATASET_PERCENTAGE=10.0
BENCH_STEPS=12
WARMUP_STEPS=4
LR=1e-5
WEIGHT_DECAY=0.01
SEED=42

ACTIVATION_CHECKPOINTING_FLAG=""
if [ "$ACTIVATION_CHECKPOINTING" = "true" ]; then
    ACTIVATION_CHECKPOINTING_FLAG="--activation_checkpointing"
fi

# ── numarun ──────────────────────────────────────────────────────────────────
if command -v numarun &>/dev/null; then
    NUMARUN="numarun"
    echo "[INFO] numarun found: enabling per-rank CPU affinity"
else
    NUMARUN=""
    echo "[INFO] numarun not found: skipping CPU affinity binding"
fi

# ── nsys profiling ───────────────────────────────────────────────────────────
PROFILE_START=$((WARMUP_STEPS + BENCH_STEPS - 3))
PROFILE_END=$((WARMUP_STEPS + BENCH_STEPS - 1))
NSYS_OUT="${OUTPUT_DIR}/${RUN_TAG}_profile"

PROFILE_FLAG=""
if [ "$PROFILE" = "true" ]; then
    PROFILE_FLAG="--profile --profile_start ${PROFILE_START} --profile_end ${PROFILE_END}"
fi

echo "Qwen3-14B PP | gbs=${GBS} mbs=${MBS} micro_batches=${MICRO_BATCHES} (derived) profile=${PROFILE}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================================"

DEEPSPEED_CMD="${NUMARUN} deepspeed --num_gpus=${GPUS_PER_NODE} ${SCRIPT_DIR}/finetune_pp.py \
    --model_name ${MODEL_NAME} \
    --lr ${LR} \
    --gbs ${GBS} \
    --micro_batch_size ${MBS} \
    --micro_batches ${MICRO_BATCHES} \
    --pp_stages ${PP_STAGES} \
    --gpus_per_node ${GPUS_PER_NODE} \
    --output_dir ${OUTPUT_DIR} \
    --run_tag ${RUN_TAG} \
    --max_length ${MAX_LENGTH} \
    --weight_decay ${WEIGHT_DECAY} \
    --seed ${SEED} \
    --log_interval ${LOG_INTERVAL} \
    --dataset_name ${DATASET_NAME} \
    --dataset_percentage ${DATASET_PERCENTAGE} \
    --warmup_steps ${WARMUP_STEPS} \
    --bench_steps ${BENCH_STEPS} \
    ${ACTIVATION_CHECKPOINTING_FLAG} \
    ${PROFILE_FLAG}"

if [ "$PROFILE" = "true" ]; then
    nsys profile \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite=true \
        -o "${NSYS_OUT}" \
        ${DEEPSPEED_CMD}
else
    eval ${DEEPSPEED_CMD}
fi
