#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Fine-tuning script for language models using DeepSpeed ZeRO Stage 3.
Supports SuperOffload, ZeRO-Offload, and ZeRO-Infinity modes.
"""

import argparse
import ctypes
import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Tuple

import torch
import deepspeed
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    set_seed,
    enable_full_determinism,
)
from deepspeed import comm as dist

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_OPTIMIZER_LR = 0.001
DEFAULT_OPTIMIZER_BETAS = (0.9, 0.999)
MS_PER_SECOND = 1000
TFLOPS_DENOMINATOR = 1e12

ALPACA_INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n"
ALPACA_INPUT_TEMPLATE = "### Input:\n{input}\n\n"
ALPACA_RESPONSE_TEMPLATE = "### Response:\n{output}"

# ── nsys helper ───────────────────────────────────────────────────────────────
# Used to restrict nsys capture to specific steps via cudaProfilerApi range.
# The shell script launches with: nsys profile --capture-range=cudaProfilerApi
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
    logger = logging.getLogger("finetune_zero3")
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


# ── Utilities ─────────────────────────────────────────────────────────────────
def get_parameter_count(parameter: torch.nn.Parameter) -> int:
    return parameter.ds_numel if hasattr(parameter, "ds_tensor") else parameter.numel()


def estimate_transformer_tflops(
    seq_len: int,
    model_size: int,
    num_layers: int,
    hidden_size: int,
    use_activation_checkpointing: bool = False,
) -> float:
    """Estimate per-sample TFLOPs for a dense decoder-only transformer."""
    coefficient = 4 if use_activation_checkpointing else 3
    tflops = (
        2 * coefficient * model_size * seq_len
        + 2 * 2 * coefficient * num_layers * hidden_size * seq_len**2
    ) / TFLOPS_DENOMINATOR
    return tflops


def preprocess_alpaca_example(
    example: Dict[str, str],
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
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


def detect_moe_model(model: AutoModelForCausalLM) -> bool:
    moe_config_attrs = [
        "num_local_experts",
        "moe_layers",
        "num_experts",
        "expert_capacity",
        "router_aux_loss_coef",
    ]
    return any(hasattr(model.config, attr) for attr in moe_config_attrs)


def create_experiment_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = args.model_name.split("/")[-1]
    activation_checkpointing = 1 if args.activation_checkpointing else 0
    return (
        f"{model_name_short}_bs{args.batch_size}_seq{args.max_length}"
        f"_ac{activation_checkpointing}_T{timestamp}"
    )


def load_tokenizer(model_name: str, logger: logging.Logger) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    return tokenizer


def load_model(
    model_name: str, attn_implementation: str, logger: logging.Logger
) -> AutoModelForCausalLM:
    logger.debug(f"Loading model: {model_name}")
    if model_name.startswith("/") and not os.path.isdir(model_name):
        raise ValueError(
            f"Local model path does not exist: '{model_name}'\n"
            "Make sure $MODELS_PATH is set correctly before running the script."
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    return model


def setup_model_training(
    model: torch.nn.Module,
    use_activation_checkpointing: bool = True,
    logger: logging.Logger = None,
) -> None:
    if use_activation_checkpointing:
        if logger:
            logger.debug("Enabling gradient checkpointing...")
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )


def create_optimizer(model: AutoModelForCausalLM) -> Any:
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    return DeepSpeedCPUAdam(
        model.parameters(),
        lr=DEFAULT_OPTIMIZER_LR,
        betas=DEFAULT_OPTIMIZER_BETAS,
    )


def load_and_preprocess_dataset(
    dataset_name: str,
    dataset_percentage: float,
    tokenizer: AutoTokenizer,
    max_length: int,
    logger: logging.Logger,
) -> Tuple[Any, DataLoader]:
    logger.debug(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    original_size = len(dataset["train"])

    if dataset_percentage < 100.0:
        subset_size = int(original_size * dataset_percentage / 100.0)
        dataset["train"] = dataset["train"].select(range(subset_size))
        logger.debug(
            f"Using {dataset_percentage}% of dataset: {subset_size}/{original_size} examples"
        )
    else:
        logger.debug(f"Using full dataset: {original_size} examples")

    tokenized_dataset = dataset["train"].map(
        lambda x: preprocess_alpaca_example(x, tokenizer, max_length),
        batched=False,
        desc="Tokenizing",
    )

    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=1,
        collate_fn=default_data_collator,
        shuffle=True,
    )
    return tokenized_dataset, train_dataloader


def initialize_wandb(
    args: argparse.Namespace, exp_name: str, logger: logging.Logger
) -> None:
    if args.use_wandb and dist.get_rank() == 0:
        try:
            wandb_run_name = args.wandb_run_name if args.wandb_run_name else exp_name
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                tags=args.wandb_tags,
                config=vars(args),
            )
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            args.use_wandb = False


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    logger = setup_logger(rank=0, log_level=args.log_level)

    exp_name = create_experiment_name(args)
    logger.debug(f"Starting experiment: {exp_name}")
    logger.debug(f"  Model: {args.model_name}")
    logger.debug(f"  Batch size: {args.batch_size}")
    logger.debug(f"  Max length: {args.max_length}")

    tokenizer = load_tokenizer(args.model_name, logger)
    model = load_model(args.model_name, args.attn_implementation, logger)

    if args.leaf_module:
        from deepspeed.utils import set_z3_leaf_modules

        logger.debug(f"Setting leaf_module to: {args.leaf_module}")
        set_z3_leaf_modules(model, [args.leaf_module])

    setup_model_training(model, args.activation_checkpointing, logger)
    optimizer = create_optimizer(model)

    tokenized_dataset, train_dataloader = load_and_preprocess_dataset(
        args.dataset_name, args.dataset_percentage, tokenizer, args.max_length, logger
    )

    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        training_data=tokenized_dataset,
        collate_fn=default_data_collator,
    )

    logger = setup_logger(rank=dist.get_rank(), log_level=args.log_level)
    initialize_wandb(args, exp_name, logger)
    model_engine.train()

    sequence_length = args.max_length
    model_size = sum(get_parameter_count(p) for p in model.parameters())
    is_moe_model = detect_moe_model(model)

    logger.debug(f"Model type: {'MoE' if is_moe_model else 'Dense'}")
    logger.debug(f"Model size: {model_size:,} parameters")

    total_tflops = None
    if not is_moe_model:
        total_tflops = estimate_transformer_tflops(
            sequence_length,
            model_size,
            model.config.num_hidden_layers,
            model.config.hidden_size,
            args.activation_checkpointing,
        )

    # nsys capture range: bench steps 7-8 (global steps warmup+7 and warmup+8)
    # bench_steps are 1-indexed here; steps warmup+1 .. warmup+bench_steps
    nsys_start_step = args.warmup_steps + args.bench_steps - 2  # step 11 when warmup=4,bench=12
    nsys_stop_step = args.warmup_steps + args.bench_steps        # after step 12

    global_step = 0
    total_train_time = 0
    iter_times = []
    tflops_per_step = []  # per bench step (after warmup), for Excel export
    losses = []

    stop = False
    for epoch in range(args.num_train_epochs):
        logger.debug(f"Starting epoch {epoch + 1}/{args.num_train_epochs}")

        for _, batch in enumerate(train_dataloader):
            # nsys range start (step 11, i.e. bench step 7)
            if global_step == nsys_start_step:
                nsys_start()

            step_start_time = time.time()
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            actual_batch_size = batch["input_ids"].shape[0]
            tokens_in_batch = actual_batch_size * sequence_length

            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

            step_time = time.time() - step_start_time
            global_step += 1

            if global_step > args.warmup_steps:
                iter_times.append(step_time)

            losses.append(loss.item())
            total_train_time += step_time

            tokens_per_second = tokens_in_batch / step_time
            step_tflops_per_gpu = None

            if not is_moe_model and total_tflops is not None:
                step_tflops_total = args.batch_size * total_tflops / step_time
                step_tflops_per_gpu = step_tflops_total / dist.get_world_size()
                if global_step > args.warmup_steps:
                    tflops_per_step.append(round(step_tflops_per_gpu, 2))

            if global_step % args.log_interval == 0:
                avg_loss = sum(losses[-args.log_interval:]) / len(
                    losses[-args.log_interval:]
                )

                if is_moe_model:
                    log_msg = (
                        f"Step {global_step:4d} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Time: {step_time * MS_PER_SECOND:5.0f}ms"
                    )
                else:
                    log_msg = (
                        f"Step {global_step:4d} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Time: {step_time * MS_PER_SECOND:5.0f}ms | "
                        f"TFLOPS/GPU: {step_tflops_per_gpu:5.2f} | "
                        f"Tokens/s: {tokens_per_second:6.0f}"
                    )

                logger.info(log_msg)

                if args.use_wandb and dist.get_rank() == 0:
                    log_dict = {
                        "train/loss": avg_loss,
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                        "perf/step_time_ms": step_time * MS_PER_SECOND,
                        "perf/tokens_per_second": tokens_per_second,
                    }
                    if step_tflops_per_gpu is not None:
                        log_dict["perf/tflops_per_gpu"] = step_tflops_per_gpu
                    wandb.log(log_dict, step=global_step)

            # nsys range stop (after step 12, i.e. bench step 8)
            if global_step == nsys_stop_step:
                nsys_stop()

            stop = global_step >= args.warmup_steps + args.bench_steps
            if stop:
                break

        if stop:
            break

    if args.save_checkpoint and dist.get_rank() == 0:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            model_engine.save_checkpoint(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    if args.use_wandb and dist.get_rank() == 0:
        try:
            wandb.finish()
        except Exception as e:
            logger.error(f"Error finishing WandB: {e}")

    # Save benchmark results as JSON.
    if dist.get_rank() == 0 and iter_times:
        avg_time = sum(iter_times) / len(iter_times)
        avg_tflops_total = (
            args.batch_size * total_tflops / avg_time if total_tflops else None
        )
        avg_tflops_per_gpu = (
            avg_tflops_total / dist.get_world_size() if avg_tflops_total else None
        )
        avg_tokens_per_s = args.batch_size * sequence_length / avg_time

        # Infer mode from DS config filename (e.g. "...superoffload_config.json")
        mode_label = "zero3"
        if args.deepspeed_config:
            for candidate in ("superoffload", "zerooffload", "zeroinfinity"):
                if candidate in args.deepspeed_config:
                    mode_label = candidate
                    break

        results = {
            "mode": mode_label,
            "model": args.model_name,
            "config_label": args.config_label,
            "batch_size": args.batch_size,
            "seq_len": sequence_length,
            "gpus": dist.get_world_size(),
            "activation_checkpointing": args.activation_checkpointing,
            "avg_tflops_per_gpu": round(avg_tflops_per_gpu, 2)
            if avg_tflops_per_gpu
            else None,
            "avg_iter_time_ms": round(avg_time * MS_PER_SECOND, 1),
            "avg_tokens_per_second": round(avg_tokens_per_s, 0),
            "tflops_per_step": tflops_per_step,
            "warmup_steps": args.warmup_steps,
            "bench_steps": args.bench_steps,
        }

        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    logger.debug("Training completed successfully!")


# ── Argument parser ───────────────────────────────────────────────────────────
def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune language models with DeepSpeed ZeRO Stage 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Micro-batch size per GPU")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_label", type=str, default="",
                        help="Human-readable config label stored in results.json")

    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--leaf_module", type=str, default=None,
                        help="Set leaf_module for MoE models (e.g. Qwen3MoeSparseMoeBlock)")
    parser.add_argument("--activation_checkpointing", action="store_true")

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.01)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--warmup_steps", type=int, default=4,
                        help="Warmup steps excluded from timing")
    parser.add_argument("--bench_steps", type=int, default=12,
                        help="Benchmark steps to measure")

    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="superoffload")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=[])

    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--dataset_percentage", type=float, default=10.0)

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not 0 < args.dataset_percentage <= 100:
        raise ValueError("dataset_percentage must be between 0 and 100")
    if args.max_length <= 0:
        raise ValueError("max_length must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.lr <= 0:
        raise ValueError("learning rate must be positive")
    if args.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive")


if __name__ == "__main__":
    parser = create_argument_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    validate_arguments(args)

    if args.deterministic:
        enable_full_determinism(args.seed)
        torch.backends.cudnn.benchmark = False
        logging.basicConfig(level=getattr(logging, args.log_level.upper()))
        logging.info("Enabled deterministic mode")
    else:
        set_seed(args.seed)
        logging.basicConfig(level=getattr(logging, args.log_level.upper()))
        logging.info(f"Set random seed to {args.seed}")

    main(args)
