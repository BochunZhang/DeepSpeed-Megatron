#!/usr/bin/env python3
"""
Fine-tuning script for language models using DeepSpeed ZeRO Stage 3.
"""

import argparse
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

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
    enable_full_determinism
)
from deepspeed import comm as dist

import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# nsys profiler control via cudaProfilerApi range.
# Active only when the script is launched with:
#   nsys profile --capture-range=cudaProfilerApi ...
# and --profile is passed to this script.
def nsys_start():
    if torch.cuda.is_available():  #ignore-cuda
        torch.cuda.cudart().cudaProfilerStart()  #ignore-cuda


def nsys_stop():
    if torch.cuda.is_available():  #ignore-cuda
        torch.cuda.cudart().cudaProfilerStop()  #ignore-cuda


def setup_logger(rank: int = 0, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("finetune_zero3")
    logger.handlers.clear()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if rank == 0:
        handler = logging.StreamHandler()
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class TrainingLogWriter:
    """Per-iteration training log writer (rank-0 only).

    Streams one JSON record per iteration to ``{run_tag}_training_log.jsonl``
    during training and, on ``close_and_convert``, rewrites the collected
    records into ``{run_tag}_training_log.json`` as a single JSON array for
    downstream tooling. Embedding ``run_tag`` in the filename keeps outputs
    identifiable when files from multiple runs are aggregated together.
    """

    def __init__(self, output_dir: str, run_tag: str = "run") -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.run_tag = run_tag
        self.jsonl_path = os.path.join(output_dir, f"{run_tag}_training_log.jsonl")
        self.json_path = os.path.join(output_dir, f"{run_tag}_training_log.json")
        self._f = open(self.jsonl_path, "w")
        self._closed = False

    def append(self, record: Dict[str, Any]) -> None:
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close_and_convert(self) -> None:
        # Idempotent: safe to call from both the normal path and a finally block.
        if self._closed:
            return
        self._closed = True
        self._f.close()

        records = []
        with open(self.jsonl_path, "r") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        with open(self.json_path, "w") as fout:
            json.dump(records, fout, indent=2)

# Constants
DEFAULT_OPTIMIZER_LR = 0.001
DEFAULT_OPTIMIZER_BETAS = (0.9, 0.999)
BYTES_TO_GB = 1e9
MS_PER_SECOND = 1000
TFLOPS_DENOMINATOR = 1e12

# Alpaca dataset formatting
ALPACA_INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n"
ALPACA_INPUT_TEMPLATE = "### Input:\n{input}\n\n"
ALPACA_RESPONSE_TEMPLATE = "### Response:\n{output}"

def get_parameter_count(parameter: torch.nn.Parameter) -> int:
    return parameter.ds_numel if hasattr(parameter, "ds_tensor") else parameter.numel()


def estimate_transformer_tflops(
    seq_len: int,
    model_size: int,
    num_layers: int,
    hidden_size: int,
    use_activation_checkpointing: bool = False
) -> float:
    """
    Estimate TFLOPS for decoder-only densde models.
    """
    coefficient = 4 if use_activation_checkpointing else 3
    tflops = (
        2 * coefficient * model_size * seq_len
        + 2 * 2 * coefficient * num_layers * hidden_size * seq_len**2
    ) / TFLOPS_DENOMINATOR
    return tflops


def preprocess_alpaca_example(
    example: Dict[str, str],
    tokenizer: AutoTokenizer,
    max_length: int = 2048
) -> Dict[str, Any]:
    prompt = ALPACA_INSTRUCTION_TEMPLATE.format(instruction=example['instruction'])

    if example.get("input", "").strip():
        prompt += ALPACA_INPUT_TEMPLATE.format(input=example['input'])

    prompt += ALPACA_RESPONSE_TEMPLATE.format(output=example['output'])

    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def detect_moe_model(model: AutoModelForCausalLM, model_name: str) -> bool:
    moe_config_attrs = [
        'num_local_experts', 'moe_layers', 'num_experts',
        'expert_capacity', 'router_aux_loss_coef'
    ]

    for attr in moe_config_attrs:
        if hasattr(model.config, attr):
            return True
    return False


def create_experiment_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = args.model_name.split("/")[-1]
    activation_checkpointing = 1 if args.activation_checkpointing else 0

    exp_name = (f"{model_name_short}_bs{args.batch_size}_seq{args.max_length}"
                f"_ac{activation_checkpointing}_T{timestamp}")
    return exp_name

def load_tokenizer(model_name: str, logger: logging.Logger) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    return tokenizer


def load_model(model_name: str, attn_implementation: str, logger: logging.Logger) -> AutoModelForCausalLM:
    logger.debug(f"Loading model: {model_name}")
    logger.debug(f"Attention implementation: {attn_implementation}")

    if model_name.startswith("/") and not os.path.isdir(model_name):
        raise ValueError(
            f"Local model path does not exist: '{model_name}'\n"
            "Make sure $MODELS_PATH is set correctly before running the script."
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation
    )

    return model


def setup_model_training(model: torch.nn.Module, use_activation_checkpointing: bool = True, logger: logging.Logger = None) -> None:
    if use_activation_checkpointing:
        if logger:
            logger.debug("Enabling gradient checkpointing...")
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )


def create_optimizer(model: AutoModelForCausalLM) -> Any:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    optimizer = DeepSpeedCPUAdam(
        model.parameters(),
        lr=DEFAULT_OPTIMIZER_LR,
        betas=DEFAULT_OPTIMIZER_BETAS
    )
    return optimizer


def load_and_preprocess_dataset(
    dataset_name: str,
    dataset_percentage: float,
    tokenizer: AutoTokenizer,
    max_length: int,
    logger: logging.Logger
) -> Tuple[Any, DataLoader]:
    logger.debug(f"Loading dataset: {dataset_name}")

    dataset = load_dataset(dataset_name)
    original_size = len(dataset["train"])

    if dataset_percentage < 100.0:
        subset_size = int(original_size * dataset_percentage / 100.0)
        dataset["train"] = dataset["train"].select(range(subset_size))
        logger.debug(f"Using {dataset_percentage}% of dataset: {subset_size}/{original_size} examples")
    else:
        logger.debug(f"Using full dataset: {original_size} examples")

    logger.debug("Tokenizing dataset...")

    tokenized_dataset = dataset["train"].map(
        lambda x: preprocess_alpaca_example(x, tokenizer, max_length),
        batched=False,
        desc="Tokenizing"
    )

    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=1,
        collate_fn=default_data_collator,
        shuffle=True
    )

    return tokenized_dataset, train_dataloader


def initialize_wandb(args: argparse.Namespace, exp_name: str, logger: logging.Logger) -> None:
    if args.use_wandb and dist.get_rank() == 0:
        try:
            wandb_run_name = args.wandb_run_name if args.wandb_run_name else exp_name
            logger.debug(f"Initializing WandB run: {wandb_run_name}")
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                tags=args.wandb_tags,
                config=vars(args)
            )
            logger.debug("WandB initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            args.use_wandb = False


def main(args: argparse.Namespace) -> None:
    logger = setup_logger(rank=0, log_level=args.log_level)

    exp_name = create_experiment_name(args)

    logger.debug(f"Starting experiment: {exp_name}")
    logger.debug("Training configuration:")
    logger.debug(f"  Model: {args.model_name}")
    logger.debug(f"  Batch size: {args.batch_size}")
    logger.debug(f"  Max length: {args.max_length}")
    logger.debug(f"  Learning rate: {args.lr}")
    logger.debug(f"  Epochs: {args.num_train_epochs}")
    logger.debug(f"  Activation checkpointing: {args.activation_checkpointing}")

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

    # Initialize DeepSpeed
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        training_data=tokenized_dataset,
        collate_fn=default_data_collator
    )

    logger = setup_logger(rank=dist.get_rank(), log_level=args.log_level)

    initialize_wandb(args, exp_name, logger)

    model_engine.train()

    sequence_length = args.max_length
    model_size = sum(get_parameter_count(p) for p in model.parameters())
    is_moe_model = detect_moe_model(model, args.model_name)

    logger.debug(f"Model type: {'MoE' if is_moe_model else 'Dense'}")
    logger.debug(f"Model size: {model_size:,} parameters")

    # Calculate TFLOPS only for non-MoE models
    total_tflops = None
    if not is_moe_model:
        total_tflops = estimate_transformer_tflops(
            sequence_length, model_size, model.config.num_hidden_layers,
            model.config.hidden_size, args.activation_checkpointing
        )

    global_step = 0
    total_tokens_processed = 0
    total_train_time = 0
    iter_times = []
    losses = []
    # Per-bench-step records (warmup steps excluded), written to results.json.
    step_records = []

    # Full per-iteration training log (includes warmup steps). Written only on
    # rank 0, streamed as JSON Lines during training and converted to a single
    # JSON array when the training loop exits (normally or via exception).
    training_log_writer: Optional[TrainingLogWriter] = None
    if dist.get_rank() == 0:
        training_log_writer = TrainingLogWriter(args.output_dir, run_tag=args.run_tag)

    try:
        stop = False
        for epoch in range(args.num_train_epochs):
            logger.debug(f"Starting epoch {epoch + 1}/{args.num_train_epochs}")

            for step, batch in enumerate(train_dataloader):
                # nsys capture range: start at profile_start, stop after profile_end.
                # Activated only when --profile is passed.
                if args.profile and global_step == args.profile_start:
                    nsys_start()

                step_start_time = time.time()
                batch = {k: v.to(model_engine.device) for k, v in batch.items()}

                actual_batch_size = batch['input_ids'].shape[0]
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

                total_tokens_processed += tokens_in_batch
                total_train_time += step_time

                tokens_per_second = tokens_in_batch / step_time
                step_tflops = None

                if not is_moe_model and total_tflops is not None:
                    step_tflops = args.batch_size * total_tflops / step_time / dist.get_world_size()

                if global_step % args.log_interval == 0:
                    avg_loss = sum(losses[-args.log_interval:]) / len(losses[-args.log_interval:])

                    if is_moe_model:
                        # Skip throughput metrics for MoE models
                        log_msg = (f"Step {global_step:4d} | "
                                  f"Loss: {avg_loss:.4f} | "
                                  f"Time: {step_time * MS_PER_SECOND:5.0f}ms")
                    else:
                        log_msg = (f"Step {global_step:4d} | "
                                  f"Loss: {avg_loss:.4f} | "
                                  f"Time: {step_time * MS_PER_SECOND:5.0f}ms | "
                                  f"TFLOPS/GPU: {step_tflops:5.2f} | "
                                  f"Tokens/s: {tokens_per_second:6.0f}")

                    logger.info(log_msg)

                    if args.use_wandb and dist.get_rank() == 0:
                        log_dict = {
                            "train/loss": avg_loss,
                            "train/epoch": epoch + 1,
                            "train/global_step": global_step,
                            "train/learning_rate": args.lr,
                            "perf/step_time_ms": step_time * MS_PER_SECOND,
                            "perf/tokens_per_second": tokens_per_second,
                        }

                        if not is_moe_model and step_tflops is not None:
                            log_dict["perf/tflops_per_gpu"] = step_tflops

                        wandb.log(log_dict, step=global_step)

                # Accumulate per-step record for JSON output (bench steps only).
                if global_step > args.warmup_steps:
                    record = {
                        "step": global_step,
                        "loss": round(loss.item(), 6),
                        "iter_time_ms": round(step_time * MS_PER_SECOND, 1),
                        "tokens_per_second": round(tokens_per_second, 1),
                    }
                    if step_tflops is not None:
                        record["tflops_per_gpu"] = round(step_tflops, 4)
                    step_records.append(record)

                # Full per-iteration training log (includes warmup steps).
                if training_log_writer is not None:
                    iter_record = {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "loss": round(loss.item(), 6),
                        "step_time_ms": round(step_time * MS_PER_SECOND, 1),
                        "tokens_per_second": (
                            round(tokens_per_second, 1) if not is_moe_model else None
                        ),
                        "learning_rate": args.lr,
                        "tflops_per_gpu": (
                            round(step_tflops, 4) if step_tflops is not None else None
                        ),
                    }
                    training_log_writer.append(iter_record)

                # nsys capture range: stop after profile_end step.
                if args.profile and global_step == args.profile_end:
                    nsys_stop()

                stop = global_step >= args.bench_steps
                if stop:
                    break

            if stop:
                break
    finally:
        if training_log_writer is not None:
            training_log_writer.close_and_convert()
            logger.info(f"Training log saved to {training_log_writer.json_path}")


    if args.save_checkpoint and dist.get_rank() == 0:
        try:
            logger.debug(f"Saving model to {args.output_dir}...")
            os.makedirs(args.output_dir, exist_ok=True)
            model_engine.save_checkpoint(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logger.debug("Model saved successfully!")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    if args.use_wandb and dist.get_rank() == 0:
        try:
            wandb.finish()
            logger.debug("WandB run finished successfully")
        except Exception as e:
            logger.error(f"Error finishing WandB run: {e}")

    # Save benchmark results as JSON (rank 0 only, bench steps only).
    if dist.get_rank() == 0 and iter_times:
        avg_time = sum(iter_times) / len(iter_times)
        avg_tokens_per_s = args.batch_size * sequence_length / avg_time

        # Infer mode label from the deepspeed config filename
        mode_label = "zero3"
        if args.deepspeed_config:
            for candidate in ("superoffload", "zerooffload", "zeroinfinity"):
                if candidate in args.deepspeed_config:
                    mode_label = candidate
                    break

        avg_tflops = None
        if not is_moe_model and total_tflops is not None:
            avg_tflops = round(args.batch_size * total_tflops / avg_time / dist.get_world_size(), 4)

        results = {
            "mode": mode_label,
            "model": args.model_name,
            "batch_size": args.batch_size,
            "seq_len": sequence_length,
            "gpus": dist.get_world_size(),
            "activation_checkpointing": args.activation_checkpointing,
            "warmup_steps": args.warmup_steps,
            "bench_steps": args.bench_steps,
            "avg_iter_time_ms": round(avg_time * MS_PER_SECOND, 1),
            "avg_tokens_per_second": round(avg_tokens_per_s, 1),
            "avg_tflops_per_gpu": avg_tflops,
            # One entry per bench step (warmup excluded), length = bench_steps - warmup_steps.
            "steps": step_records,
        }

        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, f"{args.run_tag}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    logger.debug("Training completed successfully!")

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Fine-tune language models with DeepSpeed ZeRO Stage 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--model_name", type=str, required=True,
                       help="HuggingFace model name or path")
    parser.add_argument("--lr", type=float, required=True,
                       help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, required=True,
                       help="Training batch size per device")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save model checkpoints")
    parser.add_argument("--run_tag", type=str, default="run",
                       help="Tag embedded in output filenames "
                            "(e.g. qwen3-14b_zero-offload_bs8-mbs1). "
                            "Defaults to 'run' when not provided.")

    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                       choices=["eager", "sdpa", "flash_attention_2"],
                       help="Attention implementation to use")
    parser.add_argument("--leaf_module", type=str, default=None,
                        help="Set leaf_module to enable fine-tuning MoE models")
    parser.add_argument("--activation_checkpointing", action="store_true",
                       help="Enable activation checkpointing to save memory")

    parser.add_argument("--num_train_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length for tokenization")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for optimization")
    parser.add_argument("--warmup", type=float, default=0.01,
                       help="Warmup ratio for learning rate schedule")

    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank passed from distributed launcher")

    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true",
                       help="Enable deterministic training for full reproducibility")

    parser.add_argument("--log_interval", type=int, default=1,
                       help="Log performance metrics every N steps")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level for controlling output verbosity")
    parser.add_argument("--warmup_steps", type=int, default=15,
                       help="Number of warmup steps for performance measurements")
    parser.add_argument("--bench_steps", type=int, default=100,
                       help="Number of benchmark steps to run")

    parser.add_argument("--save_checkpoint", action="store_true",
                       help="Save model checkpoint after training")

    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="superoffload",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=[],
                       help="WandB tags for the run")

    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca",
                       help="HuggingFace dataset name")
    parser.add_argument("--dataset_percentage", type=float, default=100.0,
                       help="Percentage of dataset to use (1.0-100.0)")

    parser.add_argument("--profile", action="store_true",
                       help="Enable nsys profiling via cudaProfilerApi range capture")
    parser.add_argument("--profile_start", type=int, default=0,
                       help="Global step at which to call cudaProfilerStart (inclusive)")
    parser.add_argument("--profile_end", type=int, default=0,
                       help="Global step at which to call cudaProfilerStop (inclusive)")

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.dataset_percentage <= 0 or args.dataset_percentage > 100:
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
        logging.info("Enabled deterministic mode for full reproducibility")
    else:
        set_seed(args.seed)
        logging.basicConfig(level=getattr(logging, args.log_level.upper()))
        logging.info(f"Set random seed to {args.seed}")

    main(args)
