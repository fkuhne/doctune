"""report_synthesis_spend.py - Summarize token and cost usage from synthesis JSONL.

Reads one synthesis JSONL file or a directory containing synthesis_*.jsonl files,
then reports aggregate and per-model token/cost usage.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from doctune.utils.pricing import compute_model_usage_cost


@dataclass
class _SpendTotals:  # pylint: disable=too-many-instance-attributes
    files_scanned: int = 0
    records_scanned: int = 0
    tuples_scanned: int = 0
    tuples_with_metadata: int = 0
    tuples_missing_metadata: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def _resolve_inputs(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("synthesis_*.jsonl"))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _safe_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def summarize_spend(input_path: str) -> tuple[_SpendTotals, dict[str, _SpendTotals]]:
    files = _resolve_inputs(input_path)
    totals = _SpendTotals(files_scanned=len(files))
    by_model: dict[str, _SpendTotals] = defaultdict(_SpendTotals)

    for file_path in files:
        for record in _iter_jsonl(file_path):
            totals.records_scanned += 1
            for result in record.get("results", []):
                totals.tuples_scanned += 1
                metadata = result.get("metadata")
                if not isinstance(metadata, dict):
                    totals.tuples_missing_metadata += 1
                    continue

                totals.tuples_with_metadata += 1
                model = str(metadata.get("model") or "unknown")
                model_totals = by_model[model]
                model_totals.tuples_with_metadata += 1

                input_tokens = _safe_int(metadata.get("input_tokens"))
                output_tokens = _safe_int(metadata.get("output_tokens"))
                cost_usd = _safe_float(metadata.get("cost_usd"))
                if cost_usd <= 0.0:
                    cost_usd = compute_model_usage_cost(
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

                totals.input_tokens += input_tokens
                totals.output_tokens += output_tokens
                totals.cost_usd += cost_usd

                model_totals.input_tokens += input_tokens
                model_totals.output_tokens += output_tokens
                model_totals.cost_usd += cost_usd

    return totals, dict(by_model)


def _print_summary(totals: _SpendTotals, by_model: dict[str, _SpendTotals]) -> None:
    print("============================================================")
    print("SYNTHESIS SPEND REPORT")
    print("============================================================")
    print(f"Files scanned: {totals.files_scanned}")
    print(f"Records scanned: {totals.records_scanned}")
    print(f"Tuples scanned: {totals.tuples_scanned}")
    print(f"Tuples with metadata: {totals.tuples_with_metadata}")
    print(f"Tuples missing metadata: {totals.tuples_missing_metadata}")
    print("------------------------------------------------------------")
    print(f"Total input tokens: {totals.input_tokens}")
    print(f"Total output tokens: {totals.output_tokens}")
    print(f"Total estimated cost (USD): ${totals.cost_usd:.6f}")

    if not by_model:
        print("------------------------------------------------------------")
        print("No per-model breakdown available (no metadata found).")
        return

    print("------------------------------------------------------------")
    print("Per-model breakdown:")
    for model, model_totals in sorted(
        by_model.items(), key=lambda item: item[1].cost_usd, reverse=True,
    ):
        print(
            f"- {model}: tuples={model_totals.tuples_with_metadata}, "
            f"input={model_totals.input_tokens}, output={model_totals.output_tokens}, "
            f"cost=${model_totals.cost_usd:.6f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize token and cost usage from synthesis JSONL cache data",
    )
    parser.add_argument(
        "--input",
        default=".cache",
        help="Path to a synthesis JSONL file or directory containing synthesis_*.jsonl files",
    )
    args = parser.parse_args()

    report_totals, report_by_model = summarize_spend(args.input)
    _print_summary(report_totals, report_by_model)
