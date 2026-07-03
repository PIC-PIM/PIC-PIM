from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def percentile(values: list[int], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def scan_trace(trace_path: Path) -> list[dict]:
    last_request_by_record: dict[str, dict] = {}
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            request = json.loads(line)
            record_id = request["record_id"]
            request_ordinal = int(request["request_ordinal"])
            ai_turn_index = int(request["ai_turn_index"])
            context_tokens = 0
            for slot in request.get("slots", []):
                for segment in slot.get("segments", []):
                    context_tokens += int(segment["token_length"])
            current = {
                "record_id": record_id,
                "logical_trace_id": request.get("logical_trace_id", ""),
                "request_id": request["request_id"],
                "request_ordinal": request_ordinal,
                "ai_turn_index": ai_turn_index,
                "context_tokens": context_tokens,
            }
            previous = last_request_by_record.get(record_id)
            if previous is None or (request_ordinal, ai_turn_index) > (
                int(previous["request_ordinal"]),
                int(previous["ai_turn_index"]),
            ):
                last_request_by_record[record_id] = current
    return [last_request_by_record[key] for key in sorted(last_request_by_record)]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(last_requests: list[dict], output_png: Path) -> dict:
    original_values = [int(row["context_tokens"]) for row in last_requests]
    low_cut = percentile(original_values, 0.01)
    high_cut = percentile(original_values, 0.99)
    values = [value for value in original_values if low_cut <= value <= high_cut]
    avg = mean(values)
    p90 = percentile(values, 0.90)
    p95 = percentile(values, 0.95)
    bin_width = 2500
    max_x = math.ceil(max(values) / bin_width) * bin_width
    bins = list(range(0, max_x + bin_width, bin_width))
    weights = [100 / len(values)] * len(values)

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    fig.suptitle("SWE-agent 400 records: final prefill context length distribution", fontsize=14.5, y=1.03)
    fig.text(0.5, 0.955, "Final LLM request per trajectory; P1-P99 range shown.", ha="center", fontsize=9.8, color="0.28")
    ax.hist(values, bins=bins, weights=weights, color="#5B84B1", edgecolor="white", linewidth=0.9)
    ax.axvline(avg, color="#E6862E", linestyle=(0, (4, 2)), linewidth=2.2)
    ax.axvline(p90, color="#777777", linestyle="--", linewidth=1.8)
    ax.axvline(p95, color="#C44E52", linestyle="--", linewidth=1.8)
    ax.text(
        0.03,
        0.95,
        f"Mean {avg/1000:.1f}K\nP90 {p90/1000:.1f}K\nP95 {p95/1000:.1f}K",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.92},
    )
    ax.set_title("Final request context length")
    ax.set_xlabel("Input context tokens")
    ax.set_ylabel("Records (%)")
    ax.set_yscale("function", functions=(lambda y: y**0.5, lambda y: y**2))
    ax.set_yticks([1, 2, 5, 10, 20, 35])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:g}%"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: "0" if x == 0 else f"{int(x / 1000)}K"))
    ax.set_ylim(0, 35)
    ax.set_xlim(0, max_x)
    ax.grid(axis="y", color="0.88")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.5,
        -0.015,
        f"Showing P1-P99 range: {len(values)}/{len(original_values)} records kept, {len(original_values) - len(values)} extremes excluded.",
        ha="center",
        fontsize=9.0,
        color="0.35",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "original_records": len(original_values),
        "kept_records": len(values),
        "trimmed_records": len(original_values) - len(values),
        "trim_low_cut": low_cut,
        "trim_high_cut": high_cut,
        "mean": avg,
        "median": median(values),
        "p90": p90,
        "p95": p95,
        "min": min(values),
        "max": max(values),
    }


def write_summary_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_jsonl", default="trace.jsonl")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    last_requests = scan_trace(Path(args.trace_jsonl))
    write_csv(output_dir / "csv" / "final_request_context_lengths.csv", last_requests)
    summary = plot(last_requests, output_dir / "figures" / "final_request_context_length_distribution.png")
    write_summary_csv(output_dir / "csv" / "final_request_context_length_distribution_summary.csv", summary)
    print(f"wrote {output_dir / 'csv' / 'final_request_context_lengths.csv'}")
    print(f"wrote {output_dir / 'figures' / 'final_request_context_length_distribution.png'}")


if __name__ == "__main__":
    main()
