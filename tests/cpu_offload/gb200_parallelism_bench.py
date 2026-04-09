# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""
Benchmark: Pipeline Parallel vs ZeRO-Infinity vs SuperOffload on GB200.

Uses megatron-core GPTModel (from third_party/Megatron-v0.16.1) as the
model backend. Architecture parameters match DeepSeek-V3 dense transformer
(MoE removed since DeepSpeed Stage 3 does not support it).

Usage:
    deepspeed --num_gpus 4 tests/cpu_offload/gb200_parallelism_bench.py \
        --mode pp --mbs 1
    deepspeed --num_gpus 4 tests/cpu_offload/gb200_parallelism_bench.py \
        --mode zero_infinity --mbs 4
    deepspeed --num_gpus 4 tests/cpu_offload/gb200_parallelism_bench.py \
        --mode super_offload --mbs 4 --ratio 0.8
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# Add megatron-core to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_MEGATRON_PATH = _REPO_ROOT / "third_party" / "Megatron-v0.16.1"
sys.path.insert(0, str(_MEGATRON_PATH))

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

# Defaults matching DeepSeek-V3 dense transformer (no MoE)
DEFAULT_HIDDEN_DIM = 7168
DEFAULT_NUM_HEADS = 128   # DeepSeek-V3 --num-attention-heads
DEFAULT_FFN_HIDDEN = 18432
DEFAULT_NUM_LAYERS = 61
DEFAULT_SEQ_LEN = 4096
DEFAULT_VOCAB_SIZE = 102400
WARMUP_STEPS_DEFAULT = 5
MEASURE_STEPS_DEFAULT = 20


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_gpt_model(args, pp_size=1):
    """Build a megatron-core GPTModel with the given configuration."""
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pre_process = (pp_rank == 0)
    post_process = (pp_rank == pp_size - 1)

    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_dim,
        num_attention_heads=args.num_heads,
        ffn_hidden_size=args.ffn_hidden,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        init_method_std=0.02,
        normalization='RMSNorm',
        add_bias_linear=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp_size,
    )

    layer_spec = get_gpt_layer_local_spec()

    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=args.vocab_size,
        max_sequence_length=args.seq_len,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=False,
        position_embedding_type='rope',
        rotary_base=10000,
    )

    return model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_token_dataset(vocab_size, seq_len, total_samples):
    """Create dataset of random token IDs and labels."""
    input_ids = torch.randint(0, vocab_size, (total_samples, seq_len))
    labels = torch.randint(0, vocab_size, (total_samples, seq_len))
    return torch.utils.data.TensorDataset(input_ids, labels)


def make_token_data_loader(batch_size, vocab_size, seq_len, total_samples):
    dataset = make_token_dataset(vocab_size, seq_len, total_samples)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


class RepeatingLoader:
    """Wraps a DataLoader to restart automatically when exhausted."""

    def __init__(self, loader):
        self.loader = loader
        self._iterator = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            return next(self._iterator)


# ---------------------------------------------------------------------------
# DeepSpeed config builders
# ---------------------------------------------------------------------------

def _flops_profiler_config(warmup_steps):
    return {
        "enabled": True,
        "profile_step": warmup_steps + 1,
        "module_depth": -1,
        "top_modules": 3,
        "detailed": False,
    }


def build_pp_config(mbs, gas, warmup_steps):
    """PP mode: ZeRO-0 + BF16. DeepSpeed only manages optimizer, Megatron handles PP."""
    return {
        "train_micro_batch_size_per_gpu": mbs,
        "gradient_accumulation_steps": gas,
        "steps_per_print": 9999999,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 1e-4},
        },
        "zero_optimization": {"stage": 0},
        "bf16": {"enabled": True},
        "wall_clock_breakdown": True,
        "flops_profiler": _flops_profiler_config(warmup_steps),
    }


def build_zero_infinity_config(mbs, gas, warmup_steps):
    return {
        "train_micro_batch_size_per_gpu": mbs,
        "gradient_accumulation_steps": gas,
        "steps_per_print": 9999999,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 1e-4},
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": True},
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": int(5e8),
            "stage3_prefetch_bucket_size": int(5e8),
            "stage3_param_persistence_threshold": int(1e6),
        },
        "bf16": {"enabled": True},
        "wall_clock_breakdown": True,
        "flops_profiler": _flops_profiler_config(warmup_steps),
    }


def build_super_offload_config(mbs, gas, ratio, warmup_steps):
    return {
        "train_micro_batch_size_per_gpu": mbs,
        "gradient_accumulation_steps": gas,
        "steps_per_print": 9999999,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 1e-4},
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
                "super_offload": True,
                "ratio": ratio,
                "cpuadam_cores_perc": 0.8,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": int(5e8),
            "stage3_prefetch_bucket_size": int(5e8),
            "stage3_param_persistence_threshold": int(1e6),
        },
        "bf16": {"enabled": True},
        "wall_clock_breakdown": True,
        "flops_profiler": _flops_profiler_config(warmup_steps),
    }


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def print0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def gather_memory_stats():
    """Return list of (current_MB, peak_MB) for each rank, gathered to rank 0."""
    current = get_accelerator().memory_allocated() / (1024**2)
    peak = get_accelerator().max_memory_allocated() / (1024**2)
    local_stats = torch.tensor(
        [current, peak], dtype=torch.float64,
        device=get_accelerator().current_device_name(),
    )
    world_size = dist.get_world_size()
    all_stats = [
        torch.zeros(2, dtype=torch.float64, device=local_stats.device)
        for _ in range(world_size)
    ]
    dist.all_gather(all_stats, local_stats)
    return [(s[0].item(), s[1].item()) for s in all_stats]


def compute_param_count(args):
    """Estimate total parameter count for the GPT model."""
    per_layer = (
        4 * args.hidden_dim * args.hidden_dim  # QKV + output projections
        + 2 * args.hidden_dim * args.ffn_hidden  # FFN up + down
        + 2 * args.hidden_dim  # 2x RMSNorm
    )
    embedding = args.vocab_size * args.hidden_dim
    output_layer = args.vocab_size * args.hidden_dim
    final_norm = args.hidden_dim
    return args.num_layers * per_layer + embedding + output_layer + final_norm


def report_results(label, args, elapsed, samples_per_step):
    stats = gather_memory_stats()
    if dist.get_rank() != 0:
        return

    time_per_step = elapsed / args.measure_steps
    samples_per_sec = samples_per_step / time_per_step
    tokens_per_sec = samples_per_sec * args.seq_len
    total_params = compute_param_count(args)

    print("=" * 70)
    print(f"Benchmark: {label}")
    print(f"Mode: {args.mode} | MBS: {args.mbs} | GAS: {args.gas}", end="")
    if args.mode == "super_offload":
        print(f" | Ratio: {args.ratio}", end="")
    print()
    print(f"Layers: {args.num_layers} | Hidden: {args.hidden_dim} | "
          f"Heads: {args.num_heads} | FFN: {args.ffn_hidden} | "
          f"SeqLen: {args.seq_len} | Vocab: {args.vocab_size}")
    print(f"Total params: {total_params / 1e9:.2f}B")
    print("-" * 70)
    print(f"Measure steps:          {args.measure_steps}")
    print(f"Total elapsed time:     {elapsed:.2f} s")
    print(f"Time per step:          {time_per_step:.4f} s")
    print(f"Samples per second:     {samples_per_sec:.2f}")
    print(f"Tokens per second:      {tokens_per_sec:.0f}")
    print(f"Samples per step:       {samples_per_step}")
    print("-" * 70)
    print("GPU Memory (per rank):")
    for rank, (cur, peak) in enumerate(stats):
        print(f"  Rank {rank}: current={cur:.1f} MB, peak={peak:.1f} MB")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Forward step for Megatron PP schedule
# ---------------------------------------------------------------------------

def get_batch_for_pp(data_iterator, device):
    """Get a batch from the data iterator and prepare for GPTModel forward."""
    batch = next(data_iterator)
    input_ids = batch[0].to(device)
    labels = batch[1].to(device)
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(input_ids)
    attention_mask = None  # GPTModel creates causal mask internally when None
    return input_ids, labels, position_ids, attention_mask


def forward_step(data_iterator, model):
    """Forward step function for Megatron's pipeline schedule.

    Returns (output_tensor, loss_func) where loss_func computes loss from output.
    """
    device = next(model.parameters()).device
    input_ids, labels, position_ids, attention_mask = get_batch_for_pp(data_iterator, device)

    output = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    def loss_func(output_tensor):
        loss = output_tensor.mean()
        return loss, {"lm_loss": loss}

    return output, loss_func


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def run_pp_benchmark(args):
    """Pipeline Parallel using Megatron-native PP schedule + DeepSpeed optimizer."""
    num_gpus = dist.get_world_size()

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=num_gpus,
    )

    model = build_gpt_model(args, pp_size=num_gpus)
    model = model.to(get_accelerator().current_device_name())

    config = build_pp_config(args.mbs, args.gas, args.warmup_steps)
    engine, _, _, _ = deepspeed.initialize(
        config=config,
        model=model,
        model_parameters=model.parameters(),
    )

    forward_backward_func = get_forward_backward_func(
        pp_size=num_gpus, vp_size=None,
    )

    total_samples = (args.warmup_steps + args.measure_steps) * args.gas * args.mbs + 100
    loader = make_token_data_loader(args.mbs, args.vocab_size, args.seq_len, total_samples)
    data_iter = RepeatingLoader(loader)

    print0(f"[PP] Megatron PP={num_gpus}, starting benchmark...")

    # Warmup
    for _ in range(args.warmup_steps):
        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iter,
            model=[engine.module],
            num_microbatches=args.gas,
            seq_length=args.seq_len,
            micro_batch_size=args.mbs,
        )
        engine.step()

    # Measure
    get_accelerator().synchronize()
    dist.barrier()
    get_accelerator().reset_peak_memory_stats()
    start = time.time()

    for _ in range(args.measure_steps):
        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iter,
            model=[engine.module],
            num_microbatches=args.gas,
            seq_length=args.seq_len,
            micro_batch_size=args.mbs,
        )
        engine.step()

    get_accelerator().synchronize()
    dist.barrier()
    elapsed = time.time() - start

    samples_per_step = args.mbs * args.gas * 1  # DP=1
    report_results(f"MegatronPP={num_gpus},DP=1,ZeRO-0", args, elapsed, samples_per_step)


def _run_zero3_training_loop(engine, args, label):
    """Shared training loop for ZeRO-Infinity and SuperOffload."""
    total_samples = (args.warmup_steps + args.measure_steps) * args.gas * args.mbs + 100
    loader = make_token_data_loader(args.mbs, args.vocab_size, args.seq_len, total_samples)
    data_iter = RepeatingLoader(loader)
    device = engine.device

    print0(f"[{label}] Starting benchmark...")

    # Warmup
    for _ in range(args.warmup_steps):
        batch = next(data_iter)
        input_ids = batch[0].to(device)
        labels = batch[1].to(device)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(input_ids)
        loss = engine(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
        )
        if isinstance(loss, torch.Tensor):
            loss = loss.mean()
        engine.backward(loss)
        engine.step()

    # Measure
    get_accelerator().synchronize()
    dist.barrier()
    get_accelerator().reset_peak_memory_stats()
    start = time.time()

    for _ in range(args.measure_steps):
        batch = next(data_iter)
        input_ids = batch[0].to(device)
        labels = batch[1].to(device)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(input_ids)
        loss = engine(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
        )
        if isinstance(loss, torch.Tensor):
            loss = loss.mean()
        engine.backward(loss)
        engine.step()

    get_accelerator().synchronize()
    dist.barrier()
    elapsed = time.time() - start

    samples_per_step = args.mbs * args.gas * dist.get_world_size()
    report_results(label, args, elapsed, samples_per_step)


def run_zero_infinity_benchmark(args):
    """ZeRO-Infinity: Stage 3 + CPU offload for params and optimizer."""
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    config = build_zero_infinity_config(args.mbs, args.gas, args.warmup_steps)

    with deepspeed.zero.Init(config_dict_or_path=config):
        model = build_gpt_model(args, pp_size=1)

    engine, _, _, _ = deepspeed.initialize(
        config=config,
        model=model,
        model_parameters=model.parameters(),
    )

    _run_zero3_training_loop(engine, args, "ZeRO-Infinity(CPU)")


def run_super_offload_benchmark(args):
    """SuperOffload: Stage 3 + async CPU optimizer with configurable ratio."""
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    config = build_super_offload_config(
        args.mbs, args.gas, args.ratio, args.warmup_steps,
    )

    with deepspeed.zero.Init(config_dict_or_path=config):
        model = build_gpt_model(args, pp_size=1)

    engine, _, _, _ = deepspeed.initialize(
        config=config,
        model=model,
        model_parameters=model.parameters(),
    )

    _run_zero3_training_loop(engine, args, f"SuperOffload(ratio={args.ratio})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="GB200 parallelism benchmark (megatron-core GPTModel)",
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["pp", "zero_infinity", "super_offload"])
    parser.add_argument("--mbs", type=int, required=True,
                        help="Micro batch size per GPU")
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num_heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--ffn_hidden", type=int, default=DEFAULT_FFN_HIDDEN)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="SuperOffload CPU offload ratio (1.0=all CPU, 0.0=all GPU)")
    parser.add_argument("--gas", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS_DEFAULT)
    parser.add_argument("--measure_steps", type=int, default=MEASURE_STEPS_DEFAULT)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Injected by deepspeed launcher")
    return parser.parse_args()


def main():
    args = parse_args()

    deepspeed.init_distributed()

    total_params = compute_param_count(args)
    print0(f"Benchmark: mode={args.mode}, mbs={args.mbs}, "
           f"layers={args.num_layers}, hidden={args.hidden_dim}, "
           f"heads={args.num_heads}, ffn={args.ffn_hidden}, "
           f"seq_len={args.seq_len}, vocab={args.vocab_size}, "
           f"params={total_params / 1e9:.2f}B")

    if args.mode == "pp":
        run_pp_benchmark(args)
    elif args.mode == "zero_infinity":
        run_zero_infinity_benchmark(args)
    elif args.mode == "super_offload":
        run_super_offload_benchmark(args)


if __name__ == "__main__":
    main()
