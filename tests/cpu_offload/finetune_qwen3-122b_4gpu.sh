#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Fine-tune Qwen3-122B-A10B (MoE) with DeepSpeed ZeRO Stage 3 on 4 GPUs.
# Uses --leaf_module Qwen3MoeSparseMoeBlock to prevent ZeRO-3 from sharding
# individual Expert parameters (avoids excessive AllGather during forward).
#
# Usage (called by run_qwen3-122b.sh or standalone):
#   bash finetune_qwen3-122b_4gpu.sh <mode> <mbs> [cpu_ratio] [grad_accum_steps]
set -eo pipefail

if [ -z "${MODELS_PATH}" ]; then
    echo "Error: MODELS_PATH environment variable is not set."
    echo "  export MODELS_PATH=/path/to/your/model/cache"
    exit 1
fi

MODE=${1:?"Usage: $0 <mode> <mbs> [cpu_ratio] [grad_accum_steps]"}
MBS=${2:?"Usage: $0 <mode> <mbs> [cpu_ratio] [grad_accum_steps]"}
CPU_RATIO=${3:-0.9}
GPUS=4
GBS=256
GAS=${4:-$((GBS / MBS / GPUS))}

if [ "$MODE" = "superoffload" ]; then
    CONFIG_LABEL="superoffload-ratio${CPU_RATIO}-mbs${MBS}"
elif [ "$MODE" = "zeroinfinity" ]; then
    CONFIG_LABEL="zeroinfinity-mbs${MBS}"
else
    echo "Error: unknown mode '$MODE'. Use: superoffload | zeroinfinity"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_NAME="${MODELS_PATH}/Qwen/Qwen3-122B-A10B"
OUTPUT_DIR="${DS_ROOT}/results/cpu_offload/qwen3-122b-a10b/${CONFIG_LABEL}"
DS_CONFIG_JSON="${OUTPUT_DIR}/ds_config.json"

mkdir -p "${OUTPUT_DIR}"

if [ "$MODE" = "superoffload" ]; then
cat > "${DS_CONFIG_JSON}" << EOF
{
    "train_batch_size": $((MBS * GAS * GPUS)),
    "train_micro_batch_size_per_gpu": ${MBS},
    "gradient_accumulation_steps": ${GAS},
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": false,
        "reduce_bucket_size": 4e8,
        "sub_group_size": 4e8,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true,
            "ratio": ${CPU_RATIO},
            "super_offload": true,
            "cpuadam_cores_perc": 0.90
        }
    },
    "wall_clock_breakdown": true
}
EOF

elif [ "$MODE" = "zeroinfinity" ]; then
cat > "${DS_CONFIG_JSON}" << EOF
{
    "train_batch_size": $((MBS * GAS * GPUS)),
    "train_micro_batch_size_per_gpu": ${MBS},
    "gradient_accumulation_steps": ${GAS},
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "reduce_bucket_size": 4e8,
        "sub_group_size": 4e8,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "wall_clock_breakdown": true
}
EOF
fi

echo "========================================================"
echo "Qwen3-122B-A10B | mode=${MODE} mbs=${MBS} cpu_ratio=${CPU_RATIO} gas=${GAS}"
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
CMD="${NUMARUN} deepspeed --num_gpus=${GPUS} ${SCRIPT_DIR}/finetune_zero3.py \
    --deepspeed_config=${DS_CONFIG_JSON} \
    --model_name ${MODEL_NAME} \
    --leaf_module Qwen3MoeSparseMoeBlock \
    --lr 1e-5 \
    --batch_size ${MBS} \
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
