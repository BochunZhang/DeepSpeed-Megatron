#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Collect all *results.json files under results/cpu_offload/ and produce a
single summary.xlsx with one sheet per model plus a cross-model summary sheet.

Supports two record schemas:
  * ZeRO-3 / offload runs (finetune_zero3.py)
      - train_batch_size stored as "batch_size"
      - per-step data under "steps" (list of dicts with step/loss/
        iter_time_ms/tokens_per_second/tflops_per_gpu)
      - file name: "{run_tag}_results.json"
  * Pipeline-parallel runs (finetune_pp.py)
      - micro_batch_size stored separately, global batch under "gbs" =
        micro_batches
      - per-step TFLOPS under "tflops_per_step" (list of floats)
      - "pipeline_bubble_fraction" and "pp_stages" / "micro_batches"
      - file name: "results.json"

Usage (from DeepSpeed-v0.18.9/):
    python tests/cpu_offload/summarize_results.py

Output:
    results/cpu_offload/summary.xlsx
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
except ImportError:
    sys.exit("openpyxl is required: pip install openpyxl")

# ── Locate results root ───────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DS_ROOT = SCRIPT_DIR.parent.parent          # DeepSpeed-v0.18.9/
RESULTS_ROOT = DS_ROOT / "results" / "cpu_offload"
OUTPUT_PATH = RESULTS_ROOT / "summary.xlsx"


def load_all_results() -> List[Dict[str, Any]]:
    """Recursively find result JSON files and parse them.

    Matches both ``results.json`` (pipeline-parallel script) and
    ``{run_tag}_results.json`` (zero3 / offload script). Training logs
    (``*_training_log.json``) are skipped.
    """
    records = []
    for path in sorted(RESULTS_ROOT.rglob("*results.json")):
        if path.name.endswith("_training_log.json"):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            data["_source"] = str(path.relative_to(DS_ROOT))
            data["_source_dir"] = path.parent.name
            records.append(data)
        except Exception as e:
            print(f"[WARN] Could not parse {path}: {e}")
    return records


def model_short(model_str: str) -> str:
    """Return the last path component as a short model name."""
    if not model_str:
        return "unknown"
    return model_str.rstrip("/").split("/")[-1]


def run_config_label(rec: Dict[str, Any]) -> str:
    """Compose a human-readable label that uniquely identifies a run.

    Falls back progressively: explicit config_label → parent directory
    name (which we now always suffix with a timestamp) → mode string.
    """
    label = rec.get("config_label")
    if label:
        return str(label)
    source_dir = rec.get("_source_dir")
    if source_dir:
        return source_dir
    return str(rec.get("mode", ""))


def is_pp_record(rec: Dict[str, Any]) -> bool:
    return rec.get("mode") == "pp" or "tflops_per_step" in rec


def effective_mbs(rec: Dict[str, Any]) -> Any:
    """Micro-batch size per GPU / pipeline stage."""
    if is_pp_record(rec):
        # PP records carry micro_batch_size explicitly.
        return rec.get("micro_batch_size")
    # zero3: derive mbs = gbs / (gas * gpus).
    gbs = rec.get("gbs")
    gas = rec.get("gradient_accumulation_steps")
    gpus = rec.get("gpus")
    if gbs and gas and gpus:
        try:
            return gbs // (int(gas) * int(gpus))
        except (TypeError, ZeroDivisionError):
            return None
    return None


def global_batch(rec: Dict[str, Any]) -> Any:
    """Global (effective) training batch size.

    Both PP and zero3 now store the global batch under the "gbs" key.
    """
    return rec.get("gbs")


# ── Styling helpers ───────────────────────────────────────────────────────────
def style_header(ws, row: int, ncols: int) -> None:
    fill = PatternFill(fill_type="solid", fgColor="4472C4")
    font = Font(bold=True, color="FFFFFF")
    for col in range(1, ncols + 1):
        cell = ws.cell(row=row, column=col)
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(horizontal="center")


def autofit(ws) -> None:
    for col in ws.columns:
        max_len = max((len(str(cell.value or "")) for cell in col), default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 2, 40)


# ── Per-step row extraction ───────────────────────────────────────────────────
def iter_step_rows(rec: Dict[str, Any]):
    """Yield per-step rows as dicts, tolerant of both schemas.

    Fields emitted: step, loss, iter_time_ms, tokens_per_second, tflops_per_gpu.
    Missing values are emitted as None so downstream code can skip/summarise.
    """
    # zero3 / offload: list[dict] under "steps".
    if isinstance(rec.get("steps"), list) and rec["steps"]:
        for s in rec["steps"]:
            yield {
                "step": s.get("step"),
                "loss": s.get("loss"),
                "iter_time_ms": s.get("iter_time_ms"),
                "tokens_per_second": s.get("tokens_per_second"),
                "tflops_per_gpu": s.get("tflops_per_gpu"),
            }
        return

    # pipeline parallel: list[float] under "tflops_per_step".
    tflops_list = rec.get("tflops_per_step")
    if isinstance(tflops_list, list) and tflops_list:
        for i, t in enumerate(tflops_list, 1):
            yield {
                "step": i,
                "loss": None,
                "iter_time_ms": None,
                "tokens_per_second": None,
                "tflops_per_gpu": t,
            }


# ── Sheets ────────────────────────────────────────────────────────────────────
def write_model_sheet(wb, model_name: str, records: List[Dict[str, Any]]) -> None:
    """One sheet per model: per-step metrics across all runs of that model."""
    ws = wb.create_sheet(title=model_name[:31])  # Excel sheet name limit

    headers = ["config", "step", "loss", "iter_time_ms", "tokens_per_second",
               "tflops_per_gpu", "avg_tflops_per_gpu", "avg_iter_time_ms",
               "avg_tokens_per_second", "mode", "mbs", "global_batch",
               "seq_len", "gpus"]
    for col, h in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=h)
    style_header(ws, 1, len(headers))

    row = 2
    for rec in records:
        config = run_config_label(rec)
        avg_tflops = rec.get("avg_tflops_per_gpu")
        avg_time = rec.get("avg_iter_time_ms")
        avg_tokens = rec.get("avg_tokens_per_second")
        mode = rec.get("mode", "")
        mbs = effective_mbs(rec)
        gbs = global_batch(rec)
        seq = rec.get("seq_len", "")
        gpus = rec.get("gpus", "")

        step_rows = list(iter_step_rows(rec))
        if step_rows:
            for step in step_rows:
                ws.cell(row=row, column=1, value=config)
                ws.cell(row=row, column=2, value=step["step"])
                ws.cell(row=row, column=3, value=step["loss"])
                ws.cell(row=row, column=4, value=step["iter_time_ms"])
                ws.cell(row=row, column=5, value=step["tokens_per_second"])
                ws.cell(row=row, column=6, value=step["tflops_per_gpu"])
                ws.cell(row=row, column=7, value=avg_tflops)
                ws.cell(row=row, column=8, value=avg_time)
                ws.cell(row=row, column=9, value=avg_tokens)
                ws.cell(row=row, column=10, value=mode)
                ws.cell(row=row, column=11, value=mbs)
                ws.cell(row=row, column=12, value=gbs)
                ws.cell(row=row, column=13, value=seq)
                ws.cell(row=row, column=14, value=gpus)
                row += 1
        else:
            # MoE or missing per-step data: write a single summary row.
            ws.cell(row=row, column=1, value=config)
            ws.cell(row=row, column=2, value="N/A")
            ws.cell(row=row, column=3, value=None)
            ws.cell(row=row, column=4, value=None)
            ws.cell(row=row, column=5, value=None)
            ws.cell(row=row, column=6, value=None)
            ws.cell(row=row, column=7, value=avg_tflops)
            ws.cell(row=row, column=8, value=avg_time)
            ws.cell(row=row, column=9, value=avg_tokens)
            ws.cell(row=row, column=10, value=mode)
            ws.cell(row=row, column=11, value=mbs)
            ws.cell(row=row, column=12, value=gbs)
            ws.cell(row=row, column=13, value=seq)
            ws.cell(row=row, column=14, value=gpus)
            row += 1

    autofit(ws)


def write_summary_sheet(wb, all_records: List[Dict[str, Any]]) -> None:
    """Cross-model summary: one row per run, with avg metrics."""
    ws = wb.create_sheet(title="Summary", index=0)

    headers = ["model", "config", "mode", "mbs", "global_batch",
               "avg_tflops_per_gpu", "avg_iter_time_ms",
               "avg_tokens_per_second", "pipeline_bubble_fraction", "gpus",
               "seq_len", "source"]
    for col, h in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=h)
    style_header(ws, 1, len(headers))

    for row, rec in enumerate(all_records, 2):
        ws.cell(row=row, column=1, value=model_short(rec.get("model", "")))
        ws.cell(row=row, column=2, value=run_config_label(rec))
        ws.cell(row=row, column=3, value=rec.get("mode", ""))
        ws.cell(row=row, column=4, value=effective_mbs(rec))
        ws.cell(row=row, column=5, value=global_batch(rec))
        ws.cell(row=row, column=6, value=rec.get("avg_tflops_per_gpu"))
        ws.cell(row=row, column=7, value=rec.get("avg_iter_time_ms"))
        ws.cell(row=row, column=8, value=rec.get("avg_tokens_per_second"))
        ws.cell(row=row, column=9, value=rec.get("pipeline_bubble_fraction"))
        ws.cell(row=row, column=10, value=rec.get("gpus"))
        ws.cell(row=row, column=11, value=rec.get("seq_len"))
        ws.cell(row=row, column=12, value=rec.get("_source", ""))

    autofit(ws)


def main() -> None:
    records = load_all_results()
    if not records:
        print(f"No *results.json found under {RESULTS_ROOT}")
        return

    print(f"Found {len(records)} result(s).")

    # Group by model.
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        key = model_short(rec.get("model", "unknown"))
        by_model.setdefault(key, []).append(rec)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # Drop default empty sheet.

    write_summary_sheet(wb, records)

    for model_name, recs in sorted(by_model.items()):
        write_model_sheet(wb, model_name, recs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
