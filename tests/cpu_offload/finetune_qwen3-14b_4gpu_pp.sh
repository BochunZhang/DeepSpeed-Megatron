#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Pipeline-Parallel fine-tuning benchmark for Qwen3-14B on 4 GPUs (no CPU offload).
#
# Usage:
#   bash finetune_qwen3-14b_4gpu_pp.sh <mbs> [global_batch_size]
#
#   mbs              : micro-batch size per pipeline stage (e.g. 1 2 4)
#   global_batch_size: default 256; micro_batches = gbs / mbs
set -eo pipefail

if [ -z "${MODELS_PATH}" ]; then
    echo "Error: MODELS_PATH environment variable is not set."
    echo "  export MODELS_PATH=/path/to/your/model/cache"
    exit 1
fi

MBS=${1:?"Usage: $0 <mbs> [global_batch_size]"}
GBS=${2:-256}
GPUS=4
PP_STAGES=4
MICRO_BATCHES=$((GBS / MBS))   # total micro-batches across the pipeline

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_REPO="Qwen/Qwen3-14B"
MODEL_NAME="${MODELS_PATH}/${MODEL_REPO}"
MODEL_SHORT="qwen3-14b"

# Download the model into MODELS_PATH if it is not already present.
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

CONFIG_LABEL="pp-mbs${MBS}-gbs${GBS}"
RUN_TAG="${MODEL_SHORT}_pp_${CONFIG_LABEL}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${DS_ROOT}/results/cpu_offload/${MODEL_SHORT}/pp/${CONFIG_LABEL}_${RUN_TIMESTAMP}"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "Qwen3-14B PP | mbs=${MBS} gbs=${GBS} micro_batches=${MICRO_BATCHES}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================================"

if command -v numarun &>/dev/null; then
    NUMARUN="numarun"
    echo "[INFO] numarun found: enabling per-rank CPU affinity"
else
    NUMARUN=""
    echo "[INFO] numarun not found: skipping CPU affinity binding"
fi

NSYS_OUT="${OUTPUT_DIR}/${RUN_TAG}_profile"
CMD="${NUMARUN} deepspeed --num_gpus=${GPUS} ${SCRIPT_DIR}/finetune_pp.py \
    --model_name ${MODEL_NAME} \
    --lr 1e-5 \
    --batch_size ${MBS} \
    --micro_batch_size ${MBS} \
    --micro_batches ${MICRO_BATCHES} \
    --pp_stages ${PP_STAGES} \
    --gpus_per_node ${GPUS} \
    --output_dir ${OUTPUT_DIR} \
    --config_label ${RUN_TAG} \
    --max_length 4096 \
    --weight_decay 0.01 \
    --seed 42 \
    --log_interval 1 \
    --dataset_name tatsu-lab/alpaca \
    --dataset_percentage 10.0 \
    --warmup_steps 4 \
    --bench_steps 12 \
    --activation_checkpointing"

eval ${CMD}
