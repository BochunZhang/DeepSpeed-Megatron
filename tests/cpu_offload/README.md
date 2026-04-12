# CPU Offload Benchmark Suite

Benchmark suite comparing CPU offloading strategies against Pipeline Parallelism (PP)
for large model fine-tuning on 4 GPUs, with per-step TFLOPS tracking and nsys profiling.

**Supported models**
- Qwen2.5-72B-Instruct (dense, 72B)
- Qwen3-122B-A10B (MoE, 122B total / 10B active)

---

## Strategies Compared

| Strategy | Script suffix | What gets offloaded | Notes |
|---|---|---|---|
| **SuperOffload** | `_4gpu.sh superoffload` | Optimizer states (async, separate process) | `ratio` controls fraction on CPU |
| **ZeRO-Infinity** | `_4gpu.sh zeroinfinity` | Optimizer states + parameters | Both offloaded to CPU |
| **Pipeline Parallel** | `_4gpu_pp.sh` | Nothing (all on GPU) | PP baseline; ZeRO stage 0 |

SuperOffload is tested with four `cpu_ratio` values: **1.0 / 0.9 / 0.5 / 0.0**
(0.0 = all optimizer states stay on GPU, equivalent to ZeRO-3 without offload).

> **Known limitation — NUMA membind disabled**
>
> The test scripts run inside a Docker container and do **not** use `numactl --membind`.
> `--membind` requires the `SYS_NICE` Linux capability which Docker does not grant by default;
> calling it causes `set_mempolicy: Operation not permitted`.
>
> As a result, CPU memory allocation falls back to Linux first-touch policy: each page is
> allocated on the NUMA node where it is first written. For a 4-GPU server with 2 sockets,
> ranks whose CPUs happen to first-touch memory on the remote socket will incur **~20–30 %
> higher CPU memory latency** on every optimizer step. This affects SuperOffload and
> ZeRO-Infinity (which move large tensors between CPU and GPU every step) more than PP
> (which barely touches CPU memory during forward/backward).
>
> To re-enable NUMA binding, add `--cap-add SYS_NICE` to the Docker run command, then
> restore `--bind_cores_to_rank` in the shell scripts (or use `numarun.sh` with
> `NUMARUN_MEMBIND=1`). The reported TFLOPS numbers in this benchmark therefore represent
> a **lower bound** for CPU-offload performance on bare-metal or capability-enabled containers.

---

## Repository Layout

```
tests/cpu_offload/
├── finetune_zero3.py              # ZeRO training script (SuperOffload / ZeRO-Infinity)
├── finetune_pp.py                 # Pipeline-Parallel training script
├── finetune_qwen2.5-72b_4gpu.sh  # Single run: Qwen2.5-72B ZeRO modes
├── finetune_qwen3-122b_4gpu.sh   # Single run: Qwen3-122B ZeRO modes
├── finetune_qwen2.5-72b_4gpu_pp.sh  # Single run: Qwen2.5-72B PP
├── finetune_qwen3-122b_4gpu_pp.sh   # Single run: Qwen3-122B PP
├── run_qwen2.5-72b.sh             # Full sweep: all modes × all mbs (Qwen2.5-72B)
├── run_qwen3-122b.sh              # Full sweep: all modes × all mbs (Qwen3-122B)
├── summarize_results.py           # Collect results.json → summary.xlsx
├── requirements.txt
└── README.md
```

Results are written to:
```
DeepSpeed-v0.18.9/results/cpu_offload/<model>/<config>/
    results.json      # benchmark metrics
    profile.nsys-rep  # nsys trace (steps 11-12 only)
    ds_config.json    # the DeepSpeed config used
```

---

## Prerequisites

```bash
pip install -r requirements.txt
pip install openpyxl          # for summarize_results.py
```

Key dependencies: `torch >= 2.5.1`, `deepspeed >= 0.17.0`, `transformers >= 4.56.1`,
`flash-attn >= 2.0.0`, `datasets`, `nsys` (NVIDIA Nsight Systems, for profiling).

Set the model base path:
```bash
export MODELS_PATH=/path/to/local/model/cache
# Models expected at:
#   $MODELS_PATH/Qwen/Qwen2.5-72B-Instruct
#   $MODELS_PATH/Qwen/Qwen3-122B-A10B
```

---

## Usage

All commands run from `DeepSpeed-v0.18.9/`.

### Full sweep (recommended)

```bash
# Qwen2.5-72B: 18 runs (12 superoffload + 3 zeroinfinity + 3 pp)
bash tests/cpu_offload/run_qwen2.5-72b.sh

# Qwen3-122B-A10B: 18 runs
bash tests/cpu_offload/run_qwen3-122b.sh

# Generate Excel summary
python tests/cpu_offload/summarize_results.py
```

### Single run

```bash
# ZeRO modes: <mode> <mbs> [cpu_ratio] [grad_accum_steps]
bash tests/cpu_offload/finetune_qwen2.5-72b_4gpu.sh superoffload 1 0.9
bash tests/cpu_offload/finetune_qwen2.5-72b_4gpu.sh zeroinfinity 2
bash tests/cpu_offload/finetune_qwen3-122b_4gpu.sh  superoffload 1 0.5

# Pipeline Parallel: <mbs> [gbs]
bash tests/cpu_offload/finetune_qwen2.5-72b_4gpu_pp.sh 1
bash tests/cpu_offload/finetune_qwen3-122b_4gpu_pp.sh  2
```

**Parameters:**

| Param | Default | Description |
|---|---|---|
| `mbs` | required | Micro-batch size per GPU |
| `cpu_ratio` | `0.9` | Fraction of optimizer states on CPU (SuperOffload only) |
| `grad_accum_steps` | `256 / mbs / 4` | Gradient accumulation steps; global batch = 256 |

### Benchmark settings (fixed in all runs)

| Setting | Value |
|---|---|
| Global batch size (gbs) | 256 |
| Warmup steps | 4 |
| Benchmark steps | 12 |
| Sequence length | 4096 |
| nsys captured steps | 11–12 (bench steps 7–8) |
| Micro-batch sizes swept | 1 / 2 / 4 |

---

## results.json Format

Each run writes a `results.json` to its output directory.

### ZeRO modes (SuperOffload / ZeRO-Infinity)

```json
{
  "mode": "superoffload",
  "model": "/path/to/Qwen/Qwen2.5-72B-Instruct",
  "config_label": "superoffload-ratio0.9-mbs1",
  "gbs": 256,
  "seq_len": 4096,
  "gpus": 4,
  "activation_checkpointing": true,
  "avg_tflops_per_gpu": 142.35,
  "avg_iter_time_ms": 2840.1,
  "avg_tokens_per_second": 5734.0,
  "tflops_per_step": [140.1, 143.2, 141.8, 144.0, 142.5,
                       141.9, 143.7, 142.6, 141.2, 143.8, 142.1, 143.5],
  "warmup_steps": 4,
  "bench_steps": 12
}
```

### Pipeline Parallel (additional fields)

```json
{
  "mode": "pp",
  "pp_stages": 4,
  "micro_batches": 256,
  "pipeline_bubble_fraction": 0.038,
  ...
}
```

### Field reference

| Field | Type | Description |
|---|---|---|
| `mode` | string | `superoffload` / `zeroinfinity` / `pp` |
| `config_label` | string | Human-readable label matching the output directory name |
| `gbs` | int | Global batch size |
| `seq_len` | int | Sequence length |
| `gpus` | int | Number of GPUs used |
| `avg_tflops_per_gpu` | float \| null | Average TFLOPs per GPU over bench steps (null for MoE) |
| `avg_iter_time_ms` | float | Average iteration time in milliseconds |
| `avg_tokens_per_second` | float | Tokens processed per second (global) |
| `tflops_per_step` | list[float] \| [] | Per bench-step TFLOPs/GPU (empty for MoE models) |
| `pipeline_bubble_fraction` | float | PP only: `(stages-1) / (stages + micro_batches - 1)` |
| `warmup_steps` | int | Steps excluded from timing |
| `bench_steps` | int | Steps included in timing (length of `tflops_per_step`) |

> **MoE note:** `avg_tflops_per_gpu` and `tflops_per_step` are `null` / `[]` for
> Qwen3-122B-A10B because the standard dense-model TFLOPS formula does not account
> for sparse routing (only a fraction of experts are active per token).

---

## Architecture

### finetune_zero3.py

```
Shell script
  └─► generates ds_config.json (mode, ratio, GAS)
  └─► nsys profile --capture-range=cudaProfilerApi deepspeed finetune_zero3.py

finetune_zero3.py
  1. AutoModelForCausalLM.from_pretrained()
     - For MoE (Qwen3-122B): set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])
       Prevents ZeRO-3 from sharding individual Expert parameters,
       avoiding excessive AllGather during Expert forward pass.
  2. DeepSpeedCPUAdam(model.parameters())
  3. deepspeed.initialize()
     - Applies ZeRO Stage 3 parameter partitioning
     - super_offload=true → SuperOffloadOptimizer_Stage3
       (spawns separate CPU Adam worker process; async step overlaps backward)
  4. Training loop: forward → backward → step
     - Step warmup+7: cudaProfilerStart()   ← nsys capture begins
     - Step warmup+8: cudaProfilerStop()    ← nsys capture ends
  5. Write results.json + tflops_per_step list
```

### finetune_pp.py

```
Pipeline stages (4 GPUs, partition_method='parameters'):

  GPU 0: EmbedLayerPipe → DecoderLayerPipe[0..N/4]
  GPU 1: DecoderLayerPipe[N/4..N/2]
  GPU 2: DecoderLayerPipe[N/2..3N/4]
  GPU 3: DecoderLayerPipe[3N/4..N] → NormHeadLayerPipe

Data between stages: (hidden_states: bfloat16, position_ids: int64)
  - hidden_states carries gradients (sent backward via P2P)
  - position_ids is non-differentiable (forwarded but not backpropagated)
  - No attention_mask: flash_attention_2 applies causal masking internally

For Qwen3-MoE: each DecoderLayerPipe wraps a full Qwen3MoeDecoderLayer
(including Qwen3MoeSparseMoeBlock). Expert dispatch is local to one stage —
no cross-stage AllToAll communication.

ZeRO stage 0 (no parameter sharding): required for PP compatibility.
Optimizer states kept on CPU via DeepSpeedCPUAdam.
```

### SuperOffload internals

```
Backward pass (GPU)
  └─► gradient ready for sub-group i
  └─► superoffload_cpu_optimizer.async_step(sub_group_i, ...)
        └─► put task on mp.SimpleQueue (shared memory, pinned)
        └─► CPU Adam worker process picks up task → runs Adam step
        └─► result back on result_queue
  └─► (meanwhile, backward continues on GPU for sub-group i+1)
  └─► main process drains result_queue during/after backward
  └─► at optimizer step boundary: wait for all async ops to complete
```

Key config knobs:

| Field | Effect |
|---|---|
| `ratio` | Fraction of sub-groups offloaded to CPU Adam worker (1.0 = all CPU) |
| `cpuadam_cores_perc` | Fraction of available CPU cores given to Adam worker process |
| `super_offload` | Enable async worker (false = synchronous CPU Adam) |

---

## DeepSpeed Config Reference

### SuperOffload

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 64,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": false,
    "reduce_bucket_size": 4e8,
    "sub_group_size": 4e8,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true,
      "ratio": 0.9,
      "super_offload": true,
      "cpuadam_cores_perc": 0.90
    }
  },
  "wall_clock_breakdown": true
}
```

### ZeRO-Infinity

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param":     { "device": "cpu", "pin_memory": true }
  }
}
```

### Pipeline Parallel

```json
{
  "zero_optimization": { "stage": 0 },
  "pipeline": {
    "activation_checkpoint_interval": 1,
    "pipe_partitioned": true,
    "grad_partitioned": true
  }
}
```

---

## nsys Profiling

Each run is wrapped with:
```bash
nsys profile \
    --capture-range=cudaProfilerApi \
    --force-overwrite=true \
    -o "${OUTPUT_DIR}/profile" \
    deepspeed ...
```

Inside the Python scripts, `cudaProfilerStart()` / `cudaProfilerStop()` are called
via `ctypes` around bench steps 7–8 (global steps 11–12):

```python
# step 11 (warmup=4, bench=12 → warmup+bench-2 = 14 → 0-indexed step 10)
if global_step == warmup_steps + bench_steps - 2:
    libcudart.cudaProfilerStart()
# after step 12
if global_step == nsys_stop_step:
    libcudart.cudaProfilerStop()
```

This keeps the `.nsys-rep` file small (only 2 steps captured) while covering
steady-state performance after the pipeline is warmed up.

Open the trace:
```bash
nsys-ui results/cpu_offload/qwen2.5-72b-instruct/superoffload-ratio0.9-mbs1/profile.nsys-rep
```
