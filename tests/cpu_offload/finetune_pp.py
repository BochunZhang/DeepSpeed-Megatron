# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Pipeline-Parallel training benchmark using DeepSpeed PP (no CPU offload).

Supports dense models (Qwen2.5-72B) and MoE models (Qwen3-122B-A10B).
For MoE models, each Qwen3MoeDecoderLayer (which contains the sparse MoE block)
is wrapped as a single pipeline stage — Expert dispatch happens entirely within
one stage with no cross-stage AllToAll.

Usage:
  deepspeed --num_gpus=4 finetune_pp.py --pp_stages 4 --micro_batches 4 \
      --model_name /path/to/model ...
"""

import argparse
import ctypes
import json
import os
import time
import logging
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from deepspeed import comm as dist
from deepspeed.pipe import PipelineModule
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Constants ─────────────────────────────────────────────────────────────────
MS_PER_SECOND = 1000
TFLOPS_DENOMINATOR = 1e12
ALPACA_INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n"
ALPACA_INPUT_TEMPLATE = "### Input:\n{input}\n\n"
ALPACA_RESPONSE_TEMPLATE = "### Response:\n{output}"

# ── nsys helper ───────────────────────────────────────────────────────────────
try:
    _libcudart = ctypes.CDLL("libcudart.so")
    _NSYS_AVAILABLE = True
except OSError:
    _libcudart = None
    _NSYS_AVAILABLE = False


def nsys_start():
    if _NSYS_AVAILABLE:
        _libcudart.cudaProfilerStart()


def nsys_stop():
    if _NSYS_AVAILABLE:
        _libcudart.cudaProfilerStop()


# ── Logger ────────────────────────────────────────────────────────────────────
def setup_logger(rank: int = 0, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("finetune_pp")
    logger.handlers.clear()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if rank == 0:
        handler = logging.StreamHandler()
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ── Pipeline Layer Wrappers ───────────────────────────────────────────────────
# Data flowing between stages: (hidden_states: bfloat16, position_ids: int64)
# Only hidden_states carries gradients; position_ids is forwarded unchanged.
# flash_attention_2 handles causal masking internally (no attention_mask needed).

class EmbedLayerPipe(nn.Module):
    """Stage 0: input_ids → (hidden_states, position_ids)."""

    def __init__(self, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return (hidden_states, position_ids)


class DecoderLayerPipe(nn.Module):
    """Middle stages: (hidden_states, position_ids) → (hidden_states, position_ids).

    Works for both dense decoder layers (Qwen2.5) and MoE decoder layers (Qwen3-MoE).
    For MoE models, the full Qwen3MoeDecoderLayer (including its Qwen3MoeSparseMoeBlock)
    runs entirely within one pipeline stage — no cross-stage Expert dispatch.
    """

    def __init__(self, decoder_layer: nn.Module):
        super().__init__()
        self.layer = decoder_layer

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, position_ids = inputs
        outputs = self.layer(
            hidden_states=hidden_states,
            attention_mask=None,    # flash_attention_2 handles causal masking
            position_ids=position_ids,
            output_attentions=False,
            use_cache=False,
        )
        # outputs[0] is hidden_states; MoE layers may return extra items (router loss etc.)
        return (outputs[0], position_ids)


class NormHeadLayerPipe(nn.Module):
    """Last stage: (hidden_states, position_ids) → logits."""

    def __init__(self, norm: nn.Module, lm_head: nn.Linear):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_states, _ = inputs
        return self.lm_head(self.norm(hidden_states))


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shift by 1 and compute cross-entropy, ignoring padding (-100)."""
    shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].contiguous().view(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def build_pipeline_model(
    hf_model: AutoModelForCausalLM,
    num_stages: int,
    activation_checkpointing: bool,
) -> PipelineModule:
    """Wrap a pretrained HF model into DeepSpeed PipelineModule.

    Works for both Qwen2.5 (dense) and Qwen3-MoE architectures — both expose
    the same .model.embed_tokens / .model.layers / .model.norm / .lm_head layout.
    """
    inner = hf_model.model
    layers = [EmbedLayerPipe(inner.embed_tokens)]
    for decoder_layer in inner.layers:
        layers.append(DecoderLayerPipe(decoder_layer))
    layers.append(NormHeadLayerPipe(inner.norm, hf_model.lm_head))

    ckpt_interval = 1 if activation_checkpointing else 0
    return PipelineModule(
        layers=layers,
        num_stages=num_stages,
        loss_fn=causal_lm_loss,
        partition_method="parameters",
        activation_checkpoint_interval=ckpt_interval,
        checkpointable_layers=["DecoderLayerPipe"],
    )


# ── Dataset ───────────────────────────────────────────────────────────────────
def preprocess_alpaca_example(
    example: Dict[str, str],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dict[str, Any]:
    prompt = ALPACA_INSTRUCTION_TEMPLATE.format(instruction=example["instruction"])
    if example.get("input", "").strip():
        prompt += ALPACA_INPUT_TEMPLATE.format(input=example["input"])
    prompt += ALPACA_RESPONSE_TEMPLATE.format(output=example["output"])

    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def pp_collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate into (input_ids, labels) tuple required by DeepSpeed PP."""
    input_ids = torch.tensor([x["input_ids"] for x in batch])
    labels = torch.tensor([x["labels"] for x in batch])
    return (input_ids, labels)


def load_dataset_for_pp(
    dataset_name: str,
    dataset_percentage: float,
    tokenizer: AutoTokenizer,
    max_length: int,
    logger: logging.Logger,
):
    logger.debug(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    original_size = len(dataset["train"])

    if dataset_percentage < 100.0:
        subset_size = int(original_size * dataset_percentage / 100.0)
        dataset["train"] = dataset["train"].select(range(subset_size))
        logger.debug(f"Using {dataset_percentage}%: {subset_size}/{original_size}")
    else:
        logger.debug(f"Using full dataset: {original_size} examples")

    tokenized_dataset = dataset["train"].map(
        lambda x: preprocess_alpaca_example(x, tokenizer, max_length),
        batched=False,
        desc="Tokenizing",
    )
    return tokenized_dataset


def estimate_transformer_tflops(
    seq_len: int,
    model_size: int,
    num_layers: int,
    hidden_size: int,
    use_activation_checkpointing: bool = False,
) -> float:
    coefficient = 4 if use_activation_checkpointing else 3
    tflops = (
        2 * coefficient * model_size * seq_len
        + 2 * 2 * coefficient * num_layers * hidden_size * seq_len**2
    ) / TFLOPS_DENOMINATOR
    return tflops


def detect_moe_model(model: AutoModelForCausalLM) -> bool:
    moe_attrs = ["num_local_experts", "moe_layers", "num_experts",
                 "expert_capacity", "router_aux_loss_coef"]
    return any(hasattr(model.config, a) for a in moe_attrs)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    logger = setup_logger(rank=0, log_level=args.log_level)
    set_seed(args.seed)

    logger.info(f"Loading model: {args.model_name}")
    logger.info(
        f"PP stages={args.pp_stages}, micro_batches={args.micro_batches}, "
        f"mbs={args.micro_batch_size}, seq_len={args.max_length}"
    )

    # Load on CPU; DeepSpeed partitions across stages after initialize().
    if args.model_name.startswith("/") and not os.path.isdir(args.model_name):
        raise ValueError(
            f"Local model path does not exist: '{args.model_name}'\n"
            "Make sure $MODELS_PATH is set correctly before running the script."
        )
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    config = hf_model.config
    config.use_cache = False

    is_moe = detect_moe_model(hf_model)
    logger.info(f"Model type: {'MoE' if is_moe else 'Dense'}")

    pp_model = build_pipeline_model(hf_model, args.pp_stages, args.activation_checkpointing)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = load_dataset_for_pp(
        args.dataset_name, args.dataset_percentage, tokenizer, args.max_length, logger
    )

    from deepspeed.ops.adam import DeepSpeedCPUAdam

    optimizer = DeepSpeedCPUAdam(
        pp_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    ds_config = {
        "train_batch_size": args.batch_size,
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": args.micro_batches,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 0},
        "pipeline": {
            "activation_checkpoint_interval": 1 if args.activation_checkpointing else 0,
            "pipe_partitioned": True,
            "grad_partitioned": True,
        },
        "wall_clock_breakdown": True,
    }

    engine, _, _, _ = deepspeed.initialize(
        model=pp_model,
        optimizer=optimizer,
        config=ds_config,
        training_data=tokenized_dataset,
        collate_fn=pp_collate_fn,
    )

    logger = setup_logger(rank=dist.get_rank(), log_level=args.log_level)

    model_size = sum(p.numel() for p in pp_model.parameters())
    per_sample_tflops = None
    if not is_moe:
        per_sample_tflops = estimate_transformer_tflops(
            args.max_length, model_size, config.num_hidden_layers,
            config.hidden_size, args.activation_checkpointing,
        )

    # PP pipeline bubble fraction: (stages-1) / (stages + micro_batches - 1)
    bubble_fraction = (args.pp_stages - 1) / (args.pp_stages + args.micro_batches - 1)

    # nsys capture: bench steps 7-8 (global steps warmup+7 and warmup+8)
    nsys_start_step = args.warmup_steps + args.bench_steps - 2
    nsys_stop_step = args.warmup_steps + args.bench_steps

    iter_times = []
    tflops_per_step = []
    losses = []
    engine.train()

    for step in range(args.warmup_steps + args.bench_steps):
        if step == nsys_start_step:
            nsys_start()

        t0 = time.time()
        loss = engine.train_batch()
        step_time = time.time() - t0

        if step >= args.warmup_steps:
            iter_times.append(step_time)

        loss_val = loss.item() if loss is not None else float("nan")
        losses.append(loss_val)

        tokens_per_second = args.batch_size * args.max_length / step_time
        step_tflops_per_gpu = None

        if per_sample_tflops is not None:
            step_tflops_per_gpu = (
                args.batch_size * per_sample_tflops / step_time / args.gpus_per_node
            )
            if step >= args.warmup_steps:
                tflops_per_step.append(round(step_tflops_per_gpu, 2))

        if (step + 1) % args.log_interval == 0 and dist.get_rank() == 0:
            tflops_str = (
                f"TFLOPS/GPU: {step_tflops_per_gpu:5.2f} | " if step_tflops_per_gpu else ""
            )
            logger.info(
                f"Step {step+1:4d} | Loss: {loss_val:.4f} | "
                f"Time: {step_time * MS_PER_SECOND:5.0f}ms | "
                f"{tflops_str}Tokens/s: {tokens_per_second:6.0f}"
            )

        if step == nsys_stop_step - 1:
            nsys_stop()

    if dist.get_rank() == 0 and iter_times:
        avg_time = sum(iter_times) / len(iter_times)
        avg_tflops_per_gpu = (
            args.batch_size * per_sample_tflops / avg_time / args.gpus_per_node
            if per_sample_tflops
            else None
        )
        avg_tokens_per_s = args.batch_size * args.max_length / avg_time

        results = {
            "mode": "pp",
            "model": args.model_name,
            "config_label": args.config_label,
            "pp_stages": args.pp_stages,
            "micro_batches": args.micro_batches,
            "batch_size": args.batch_size,
            "seq_len": args.max_length,
            "gpus": args.gpus_per_node,
            "activation_checkpointing": args.activation_checkpointing,
            "avg_tflops_per_gpu": round(avg_tflops_per_gpu, 2) if avg_tflops_per_gpu else None,
            "avg_iter_time_ms": round(avg_time * MS_PER_SECOND, 1),
            "avg_tokens_per_second": round(avg_tokens_per_s, 0),
            "tflops_per_step": tflops_per_step,
            "pipeline_bubble_fraction": round(bubble_fraction, 3),
            "warmup_steps": args.warmup_steps,
            "bench_steps": args.bench_steps,
        }

        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("=" * 60)
        if avg_tflops_per_gpu:
            logger.info(f"Avg TFLOPs/GPU : {avg_tflops_per_gpu:.2f}")
        logger.info(f"Avg iter time  : {avg_time * MS_PER_SECOND:.1f} ms")
        logger.info(f"Avg tokens/s   : {avg_tokens_per_s:.0f}")
        logger.info(f"Pipeline bubble: {bubble_fraction:.1%}")
        logger.info(f"Results saved  : {results_path}")
        logger.info("=" * 60)


# ── Argument parser ───────────────────────────────────────────────────────────
def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline-Parallel benchmark (no CPU offload baseline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Global batch size = micro_batch_size × micro_batches")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_label", type=str, default="",
                        help="Human-readable config label stored in results.json")

    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--activation_checkpointing", action="store_true")

    parser.add_argument("--pp_stages", type=int, default=4)
    parser.add_argument("--micro_batches", type=int, default=4,
                        help="Number of micro-batches per train_batch() call")
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="Samples per micro-batch per GPU")
    parser.add_argument("--gpus_per_node", type=int, default=4)

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--warmup_steps", type=int, default=4)
    parser.add_argument("--bench_steps", type=int, default=12)

    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--dataset_percentage", type=float, default=10.0)

    parser.add_argument("--local_rank", type=int, default=-1)
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.batch_size != args.micro_batch_size * args.micro_batches:
        raise ValueError(
            f"batch_size ({args.batch_size}) must equal "
            f"micro_batch_size ({args.micro_batch_size}) × micro_batches ({args.micro_batches})"
        )

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    main(args)
