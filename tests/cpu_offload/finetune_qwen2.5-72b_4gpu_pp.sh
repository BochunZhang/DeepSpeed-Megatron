#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Pipeline-Parallel benchmark for Qwen2.5-72B-Instruct on 4 GPUs (no CPU offload).
#
# Usage:
#   bash finetune_qwen2.5-72b_4gpu_pp.sh <mbs> [global_batch_size]
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

CONFIG_LABEL="pp-mbs${MBS}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_NAME="${MODELS_PATH}/Qwen/Qwen2.5-72B-Instruct"
OUTPUT_DIR="${DS_ROOT}/results/cpu_offload/qwen2.5-72b-instruct/${CONFIG_LABEL}"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "Qwen2.5-72B-Instruct PP | mbs=${MBS} micro_batches=${MICRO_BATCHES}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================================"

if command -v numarun &>/dev/null; then
    NUMARUN="numarun"
    echo "[INFO] numarun found: enabling per-rank CPU affinity"
else
    NUMARUN=""
    echo "[INFO] numarun not found: skipping CPU affinity binding"
fi

NSYS_OUT="${OUTPUT_DIR}/profile"
CMD="${NUMARUN} deepspeed --num_gpus=${GPUS} ${SCRIPT_DIR}/finetune_pp.py \
    --model_name ${MODEL_NAME} \
    --lr 1e-5 \
    --batch_size ${MBS} \
    --micro_batch_size ${MBS} \
    --micro_batches ${MICRO_BATCHES} \
    --pp_stages ${PP_STAGES} \
    --gpus_per_node ${GPUS} \
    --output_dir ${OUTPUT_DIR} \
    --config_label ${CONFIG_LABEL} \
    --max_length 4096 \
    --weight_decay 0.01 \
    --seed 42 \
    --log_interval 1 \
    --dataset_name tatsu-lab/alpaca \
    --dataset_percentage 10.0 \
    --warmup_steps 4 \
    --bench_steps 12 \
    --activation_checkpointing"

nsys profile \
    --capture-range=cudaProfilerApi \
    --force-overwrite=true \
    -o "${NSYS_OUT}" \
    ${CMD}
