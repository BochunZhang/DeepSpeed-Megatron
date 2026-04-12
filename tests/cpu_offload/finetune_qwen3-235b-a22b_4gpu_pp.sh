#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Pipeline-Parallel benchmark for Qwen3-235B-A22B-Instruct-2507 (MoE) on 4 GPUs (no CPU offload).
# Each Qwen3MoeDecoderLayer (including its sparse MoE block) is wrapped as a single
# pipeline stage — Expert computation is fully local to one stage.
#
# Usage:
#   bash finetune_qwen3-235b-a22b_4gpu_pp.sh <mbs> [global_gbs]
set -eo pipefail

if [ -z "${MODELS_PATH}" ]; then
    echo "Error: MODELS_PATH environment variable is not set."
    echo "  export MODELS_PATH=/path/to/your/model/cache"
    exit 1
fi

MBS=${1:?"Usage: $0 <mbs> [global_gbs]"}
GBS=${2:-256}
GPUS=4
PP_STAGES=4
MICRO_BATCHES=$((GBS / MBS))

CONFIG_LABEL="pp-mbs${MBS}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_NAME="${MODELS_PATH}/Qwen/Qwen3-235B-A22B-Instruct-2507"
OUTPUT_DIR="${DS_ROOT}/results/cpu_offload/qwen3-235b-a22b-instruct-2507/${CONFIG_LABEL}"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "Qwen3-235B-A22B-Instruct-2507 PP | mbs=${MBS} micro_batches=${MICRO_BATCHES}"
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
    --gbs ${MBS} \
    --micro_batch_size ${MBS} \
    --micro_batches ${MICRO_BATCHES} \
    --pp_stages ${PP_STAGES} \
    --gpus_per_node ${GPUS} \
    --output_dir ${OUTPUT_DIR} \
    --run_tag ${CONFIG_LABEL} \
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
