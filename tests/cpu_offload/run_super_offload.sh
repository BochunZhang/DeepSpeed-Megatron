#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# SuperOffload benchmark on GB200.
# Usage:
#   bash run_super_offload.sh                               # default RATIO=0.8, MBS=1
#   RATIO=1.0 MBS=4 bash run_super_offload.sh               # override ratio and mbs
#   ENABLE_NSYS=1 RATIO=0.8 bash run_super_offload.sh       # with nsys profiling
set -euo pipefail

# ======================== NSYS control ========================
ENABLE_NSYS=${ENABLE_NSYS:-0}

# ======================== Parameters ==========================
NUM_GPUS=${NUM_GPUS:-4}
HIDDEN_DIM=${HIDDEN_DIM:-7168}
NUM_HEADS=${NUM_HEADS:-128}
FFN_HIDDEN=${FFN_HIDDEN:-18432}
NUM_LAYERS=${NUM_LAYERS:-61}
SEQ_LEN=${SEQ_LEN:-4096}
MBS=${MBS:-1}
GAS=${GAS:-1}
RATIO=${RATIO:-0.8}
WARMUP=${WARMUP:-5}
MEASURE=${MEASURE:-20}

# ======================== Resolve paths =======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/gb200_parallelism_bench.py"

BENCH_ARGS="\
    --mode super_offload \
    --mbs ${MBS} \
    --gas ${GAS} \
    --ratio ${RATIO} \
    --hidden_dim ${HIDDEN_DIM} \
    --num_heads ${NUM_HEADS} \
    --ffn_hidden ${FFN_HIDDEN} \
    --num_layers ${NUM_LAYERS} \
    --seq_len ${SEQ_LEN} \
    --warmup_steps ${WARMUP} \
    --measure_steps ${MEASURE}"

DS_CMD="deepspeed --num_gpus ${NUM_GPUS} ${BENCH_SCRIPT} ${BENCH_ARGS}"

# ======================== Launch ==============================
if [ "${ENABLE_NSYS}" -eq 1 ]; then
    NSYS_OUTPUT=${NSYS_OUTPUT:-"nsys_super_offload_r${RATIO}_mbs${MBS}"}
    echo "[NSYS] Profiling enabled, output: ${NSYS_OUTPUT}"
    exec nsys profile \
        --trace=cuda,nvtx,osrt \
        --trace-fork-before-exec=true \
        --cuda-memory-usage=true \
        --output="${NSYS_OUTPUT}" \
        --force-overwrite=true \
        ${DS_CMD}
else
    exec ${DS_CMD}
fi
