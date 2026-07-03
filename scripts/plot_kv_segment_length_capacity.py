from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BUCKETS = [
    ("<=64 tok", 0, 64),
    ("65-256 tok", 65, 256),
    ("257-512 tok", 257, 512),
    ("513-1K tok", 513, 1024),
    (">1K tok", 1025, 10**18),
]


def bucket_label(tokens: int) -> str:
    for label, lo, hi in BUCKETS:
        if lo <= tokens <= hi:
            return label
    return ">1K tok"


def scan_trace(trace_path: Path) -> list[dict]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"segments": 0, "tokens": 0})
    total_segments = 0
    total_tokens = 0

    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            request = json.loads(line)
            for slot in request.get("slots", []):
                for segment in slot.get("segments", []):
                    tokens = int(segment["token_length"])
                    bucket = bucket_label(tokens)
                    stats[bucket]["segments"] += 1
                    stats[bucket]["tokens"] += tokens
                    total_segments += 1
                    total_tokens += tokens

    rows: list[dict] = []
    for label, _, _ in BUCKETS:
        row = stats[label]
        rows.append(
            {
                "length_bucket": label,
                "segment_occurrences": row["segments"],
                "segment_count_share": row["segments"] / total_segments if total_segments else 0,
                "kv_capacity_tokens": row["tokens"],
                "kv_capacity_share": row["tokens"] / total_tokens if total_tokens else 0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], output_png: Path) -> None:
    labels = [row["length_bucket"] for row in rows]
    count_values = [float(row["segment_count_share"]) * 100 for row in rows]
    capacity_values = [float(row["kv_capacity_share"]) * 100 for row in rows]
    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(
        [i - width / 2 for i in x],
        count_values,
        width=width,
        color="#4E79A7",
        edgecolor="black",
        label="Segment count",
    )
    ax.bar(
        [i + width / 2 for i in x],
        capacity_values,
        width=width,
        color="#F28E2B",
        edgecolor="black",
        label="KV capacity",
    )
    for i, value in enumerate(count_values):
        ax.text(i - width / 2, value + 0.9, f"{value:.1f}%", ha="center", fontsize=9)
    for i, value in enumerate(capacity_values):
        ax.text(i + width / 2, value + 0.9, f"{value:.1f}%", ha="center", fontsize=9)

    ax.set_title("KV segment length and capacity distribution")
    ax.set_ylabel("Share (%)")
    ax.set_xticks(list(x), labels)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="0.88")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(count_values + capacity_values) * 1.18)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_jsonl", default="trace.jsonl")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = scan_trace(Path(args.trace_jsonl))
    write_csv(output_dir / "csv" / "kv_segment_length_capacity.csv", rows)
    plot(rows, output_dir / "figures" / "kv_segment_length_capacity.png")
    print(f"wrote {output_dir / 'csv' / 'kv_segment_length_capacity.csv'}")
    print(f"wrote {output_dir / 'figures' / 'kv_segment_length_capacity.png'}")


if __name__ == "__main__":
    main()
