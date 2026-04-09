# GB200 CPU Offload Benchmark Suite

Benchmark comparing three parallelism strategies on NVIDIA GB200 (Grace-Blackwell).
Uses **megatron-core GPTModel** as the model backend.

| Strategy | Description |
|----------|-------------|
| **Pipeline Parallel** | Megatron-native PP=4, DP=1, ZeRO-0. Model layers split across 4 GPUs. |
| **ZeRO-Infinity** | ZeRO Stage 3 + CPU offload for params and optimizer states. |
| **SuperOffload** | ZeRO Stage 3 + async CPU Adam in a separate process, with configurable GPU/CPU ratio. |

## Prerequisites

```bash
# Clone the Megatron-LM submodule (required)
git submodule update --init third_party/Megatron-v0.16.1

# Install megatron-core dependencies
pip install pydantic einops
```

## Hardware Requirements

- 4x NVIDIA Blackwell GPUs (192 GB HBM each)
- 2x Grace CPUs (high-bandwidth NVLink-C2C to GPUs)

## Model Specification

megatron-core GPTModel based on DeepSeek-V3 architecture (MoE removed since DeepSpeed Stage 3 does not support it).

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_dim | 7168 | DeepSeek-V3 hidden_size |
| num_heads | 128 | DeepSeek-V3 num_attention_heads |
| ffn_hidden | 18432 | DeepSeek-V3 ffn_hidden_size |
| num_layers | 61 | DeepSeek-V3 num_layers |
| seq_len | 4096 | DeepSeek-V3 seq_length |
| vocab_size | 102400 | Approximate DeepSeek-V3 vocab |
| dtype | bf16 | |
| Total params | ~30B | Including embedding + output layer |

### Memory Estimates

| Mode | Params/GPU | Optimizer/GPU | Activations | Total |
|------|-----------|---------------|-------------|-------|
| PP=4 (~15 layers/stage) | ~14 GB | ~28 GB | ~1 GB | ~43 GB |
| ZeRO-3 (4-way shard) | ~14 GB | ~43 GB | ~3 GB | ~60 GB |

All fit within GB200's 192 GB HBM.

## Quick Start

```bash
cd tests/cpu_offload/

# Pipeline Parallel (PP=4, DP=1)
bash run_pp.sh

# ZeRO-Infinity (CPU offload, no NVMe)
bash run_zero_infinity.sh

# SuperOffload (80% optimizer on CPU, 20% on GPU)
RATIO=0.8 bash run_super_offload.sh
```

## NSYS Profiling

Set `ENABLE_NSYS=1` to wrap the launch command with `nsys profile`:

```bash
ENABLE_NSYS=1 bash run_pp.sh
ENABLE_NSYS=1 NSYS_OUTPUT=my_profile bash run_zero_infinity.sh
```

The generated `.nsys-rep` file can be opened in NVIDIA Nsight Systems.

## Configuration

All parameters can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_GPUS` | 4 | Number of GPUs |
| `HIDDEN_DIM` | 7168 | Model hidden dimension |
| `NUM_HEADS` | 128 | Number of attention heads |
| `FFN_HIDDEN` | 18432 | FFN intermediate dimension |
| `NUM_LAYERS` | 61 | Number of transformer layers |
| `SEQ_LEN` | 4096 | Sequence length |
| `MBS` | 1 | Micro-batch size per GPU |
| `GAS` | 1 | Gradient accumulation steps |
| `RATIO` | 0.8 | SuperOffload CPU ratio (1.0=all CPU, 0.0=all GPU) |
| `WARMUP` | 5 | Warmup steps (not measured) |
| `MEASURE` | 20 | Steps to measure |
| `ENABLE_NSYS` | 0 | Set to 1 to enable nsys profiling |
| `NSYS_OUTPUT` | auto | nsys output file name (without extension) |

### Example: Sweep SuperOffload Ratios

```bash
for RATIO in 1.0 0.8 0.2 0.0; do
    for MBS in 1 2 4 8; do
        RATIO=$RATIO MBS=$MBS bash run_super_offload.sh
    done
done
```

### Example: Small-Scale Local Test

```bash
NUM_GPUS=2 HIDDEN_DIM=512 NUM_HEADS=8 FFN_HIDDEN=1024 \
    NUM_LAYERS=8 SEQ_LEN=128 bash run_pp.sh
```

## How Each Mode Works

### Pipeline Parallel
- Model layers are partitioned across GPUs (stage 0..3)
- Each GPU only holds its assigned layers
- 1F1B schedule orchestrates forward/backward across stages
- ZeRO-0: no optimizer state sharding

### ZeRO-Infinity
- Parameters sharded across all GPUs (ZeRO Stage 3)
- Both params and optimizer states offloaded to CPU (`pin_memory=True`)
- Forward: all-gather params from CPU -> GPU, compute, release
- Backward: same + reduce-scatter gradients
- Step: CPU Adam on pinned memory

### SuperOffload
- Parameters sharded across GPUs (ZeRO Stage 3)
- Optimizer states on CPU, executed by a **separate process** with dedicated CPU cores
- During backward, completed sub-groups' optimizer steps are **submitted asynchronously**
- `ratio` controls how many sub-groups go to CPU vs GPU
  - `ratio=1.0`: all on CPU (maximum overlap potential)
  - `ratio=0.8`: 80% CPU, 20% GPU (recommended for GB200)
  - `ratio=0.0`: all on GPU (no CPU offload, still uses SuperOffload code path)

## Output

The benchmark prints per-step timing, throughput, and per-GPU memory:

```
======================================================================
Benchmark: SuperOffload(ratio=0.8)
Mode: super_offload | MBS: 4 | GAS: 1 | Ratio: 0.8
Layers: 61 | Hidden: 7168 | Heads: 128 | FFN: 18432 | SeqLen: 4096 | Vocab: 102400
Total params: 30.12B
----------------------------------------------------------------------
Measure steps:          20
Total elapsed time:     45.32 s
Time per step:          2.2660 s
Samples per second:     7.06
Tokens per second:      28917
Samples per step:       16
----------------------------------------------------------------------
GPU Memory (per rank):
  Rank 0: current=12340.1 MB, peak=18560.3 MB
  ...
======================================================================
```

DeepSpeed Flops Profiler is also enabled and will print TFLOPS at `warmup_steps + 1`.

## File Structure

| File | Description |
|------|-------------|
| `gb200_parallelism_bench.py` | Core benchmark: model, configs, training loops, measurement |
| `run_pp.sh` | Launch Pipeline Parallel test |
| `run_zero_infinity.sh` | Launch ZeRO-Infinity test |
| `run_super_offload.sh` | Launch SuperOffload test |
| `README.md` | This file |
