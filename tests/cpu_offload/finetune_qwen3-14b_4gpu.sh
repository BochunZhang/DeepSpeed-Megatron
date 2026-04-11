#!/bin/bash
set -eo pipefail

echo "================================================"
echo "Qwen3-14B Fine-tuning with DeepSpeed on 4 GPU"
echo "================================================"

# ── Argument parsing ──────────────────────────────────────────────────────────
MODE=superoffload
BATCH_SIZE=8
MBS=1
GPUS_PER_NODE=4
CPU_RATIO=0.90
PROFILE=false

usage() {
    echo "Usage: $0 [--mode MODE] [--batch_size N] [--mbs N] [--cpu_ratio R] [--profile]"
    echo "  --mode       superoffload (default) | zerooffload | zeroinfinity"
    echo "  --batch_size train batch size across all GPUs (default: 8)"
    echo "  --mbs        train micro batch size per GPU (default: 1)"
    echo "               gradient_accumulation_steps is derived as:"
    echo "               batch_size / (mbs * num_gpus)"
    echo "  --cpu_ratio  CPU offload ratio for superoffload (default: 0.90)"
    echo "  --profile    enable nsys profiling (default: off)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2";       shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --mbs)        MBS="$2";        shift 2 ;;
        --cpu_ratio)  CPU_RATIO="$2";  shift 2 ;;
        --profile)    PROFILE=true;    shift   ;;
        --help|-h)    usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# ── Derive gradient accumulation steps ────────────────────────────────────────
# Per DeepSpeed: train_batch_size = mbs * gas * dp_size
DP_SIZE=${GPUS_PER_NODE}
GAS_DENOM=$(( MBS * DP_SIZE ))
if (( GAS_DENOM == 0 )) || (( BATCH_SIZE % GAS_DENOM != 0 )); then
    echo "Error: batch_size (${BATCH_SIZE}) must be divisible by mbs (${MBS}) * num_gpus (${DP_SIZE}) = ${GAS_DENOM}"
    exit 1
fi
GAS=$(( BATCH_SIZE / GAS_DENOM ))
if (( GAS < 1 )); then
    echo "Error: derived gradient_accumulation_steps (${GAS}) must be >= 1"
    exit 1
fi

# ── Validate mode ─────────────────────────────────────────────────────────────
if [ "$MODE" = "superoffload" ]; then
    MODE_LABEL="super-offload"
    CONFIG_LABEL="bs${BATCH_SIZE}-mbs${MBS}-cpu${CPU_RATIO}"
elif [ "$MODE" = "zerooffload" ]; then
    MODE_LABEL="zero-offload"
    CONFIG_LABEL="bs${BATCH_SIZE}-mbs${MBS}"
elif [ "$MODE" = "zeroinfinity" ]; then
    MODE_LABEL="zero-infinity"
    CONFIG_LABEL="bs${BATCH_SIZE}-mbs${MBS}"
else
    echo "Error: Unknown mode '$MODE'. Use: superoffload | zerooffload | zeroinfinity"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${MODELS_PATH}" ]; then
    echo "Error: MODELS_PATH environment variable is not set."
    echo "  export MODELS_PATH=/path/to/your/model/cache"
    exit 1
fi

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

# Tag that is embedded in every output filename so runs remain identifiable
# even when files from several runs are aggregated together.
RUN_TAG="${MODEL_SHORT}_${MODE_LABEL}_${CONFIG_LABEL}"

# Append a timestamp to the output directory itself so repeated runs with the
# same (mode, batch_size, mbs) do not silently overwrite earlier results.
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${DS_ROOT}/results/cpu_offload/${MODEL_SHORT}/${MODE_LABEL}/${CONFIG_LABEL}_${RUN_TIMESTAMP}"
DS_CONFIG_JSON="${OUTPUT_DIR}/${RUN_TAG}_ds_config.json"

mkdir -p "${OUTPUT_DIR}"

# ── Script argument parameters ────────────────────────────────────────────────
ACTIVATION_CHECKPOINTING=true
SAVE_CHECKPOINT=false
MAX_LENGTH=4096
LOG_INTERVAL=1
DATASET_NAME="tatsu-lab/alpaca"
DATASET_PERCENTAGE=10.0
USE_WANDB=false
WANDB_PROJECT="qwen3-14b"
WANDB_RUN_NAME="qwen3-14b-${MODE}"
DETERMINISTIC=false
BENCH_STEPS=10
WARMUP_STEPS=20

EPOCHS=1
LR=1e-5
WARMUP=0.05
WEIGHT_DECAY=0.01
SEED=42

ACTIVATION_CHECKPOINTING_FLAG=""
if [ "$ACTIVATION_CHECKPOINTING" = "true" ]; then
    ACTIVATION_CHECKPOINTING_FLAG="--activation_checkpointing"
fi

SAVE_CHECKPOINT_ARG=""
if [ "$SAVE_CHECKPOINT" = "true" ]; then
    SAVE_CHECKPOINT_ARG="--save_checkpoint"
fi

WANDB_FLAG=""
if [ "$USE_WANDB" = "true" ]; then
    WANDB_FLAG="--use_wandb"
fi

DETERMINISTIC_FLAG=""
if [ "$DETERMINISTIC" = "true" ]; then
    DETERMINISTIC_FLAG="--deterministic"
fi

# ── DeepSpeed configuration ───────────────────────────────────────────────────
if [ "$MODE" = "superoffload" ]; then
cat > "${DS_CONFIG_JSON}" << EOF
{
    "train_batch_size": ${BATCH_SIZE},
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

elif [ "$MODE" = "zerooffload" ]; then
cat > "${DS_CONFIG_JSON}" << EOF
{
    "train_batch_size": ${BATCH_SIZE},
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
            "pin_memory": true
        }
    },
    "wall_clock_breakdown": true
}
EOF

elif [ "$MODE" = "zeroinfinity" ]; then
cat > "${DS_CONFIG_JSON}" << EOF
{
    "train_batch_size": ${BATCH_SIZE},
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

# ── numarun ───────────────────────────────────────────────────────────────────
if command -v numarun &>/dev/null; then
    NUMARUN="numarun"
    echo "[INFO] numarun found: enabling per-rank CPU affinity"
else
    NUMARUN=""
    echo "[INFO] numarun not found: skipping CPU affinity binding"
fi

# ── nsys profiling ────────────────────────────────────────────────────────────
# Capture rounds 6 and 7 out of 10 total steps.
# profile_start: global_step value BEFORE the 6th step's forward pass (= 5)
# profile_end:   global_step value AFTER  the 7th step completes        (= 7)
PROFILE_START=5
PROFILE_END=7
NSYS_OUT="${OUTPUT_DIR}/${RUN_TAG}_profile"

PROFILE_FLAG=""
if [ "$PROFILE" = "true" ]; then
    PROFILE_FLAG="--profile --profile_start ${PROFILE_START} --profile_end ${PROFILE_END}"
fi

echo "Qwen3-14B | mode=${MODE} batch_size=${BATCH_SIZE} mbs=${MBS} gas=${GAS} (derived) cpu_ratio=${CPU_RATIO} profile=${PROFILE}"
echo "Output: ${OUTPUT_DIR}"
echo "================================================"

DEEPSPEED_CMD="${NUMARUN} deepspeed --num_gpus=${GPUS_PER_NODE} ${SCRIPT_DIR}/finetune_zero3.py \
    --deepspeed_config=${DS_CONFIG_JSON} \
    --model_name ${MODEL_NAME} \
    --num_train_epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --weight_decay ${WEIGHT_DECAY} \
    --output_dir ${OUTPUT_DIR} \
    --run_tag ${RUN_TAG} \
    --seed ${SEED} \
    --max_length ${MAX_LENGTH} \
    --log_interval ${LOG_INTERVAL} \
    --dataset_name ${DATASET_NAME} \
    --dataset_percentage ${DATASET_PERCENTAGE} \
    --bench_steps ${BENCH_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    ${ACTIVATION_CHECKPOINTING_FLAG} \
    ${SAVE_CHECKPOINT_ARG} \
    ${WANDB_FLAG} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME} \
    ${DETERMINISTIC_FLAG} \
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
