from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def scan_trace(trace_path: Path) -> list[dict]:
    prev_hash_by_record: dict[str, set[str]] = defaultdict(set)
    prev_position_by_record: dict[str, set[tuple[str, int, int]]] = defaultdict(set)

    total_tokens = 0
    strict_tokens = 0
    shifted_tokens = 0
    cold_tokens = 0

    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            request = json.loads(line)
            record_id = request["record_id"]
            prev_hashes = prev_hash_by_record[record_id]
            prev_positions = prev_position_by_record[record_id]
            current_hashes: set[str] = set()
            current_positions: set[tuple[str, int, int]] = set()

            for slot in request.get("slots", []):
                for segment in slot.get("segments", []):
                    text_hash = segment["text_hash"]
                    tokens = int(segment["token_length"])
                    start = int(segment["start"])
                    end = int(segment["end"])
                    position_key = (text_hash, start, end)

                    content_reusable = text_hash in prev_hashes
                    strict_position_reusable = position_key in prev_positions
                    shifted_pic_only = content_reusable and not strict_position_reusable
                    cold_new = not content_reusable

                    total_tokens += tokens
                    strict_tokens += tokens if strict_position_reusable else 0
                    shifted_tokens += tokens if shifted_pic_only else 0
                    cold_tokens += tokens if cold_new else 0
                    current_hashes.add(text_hash)
                    current_positions.add(position_key)

            prev_hash_by_record[record_id] = current_hashes
            prev_position_by_record[record_id] = current_positions

    content_tokens = strict_tokens + shifted_tokens
    rows = [
        {
            "reuse_type": "content_reuse_total",
            "tokens": content_tokens,
            "token_share": content_tokens / total_tokens if total_tokens else 0,
        },
        {
            "reuse_type": "strict_position_reuse",
            "tokens": strict_tokens,
            "token_share": strict_tokens / total_tokens if total_tokens else 0,
        },
        {
            "reuse_type": "position_shifted_pic_only",
            "tokens": shifted_tokens,
            "token_share": shifted_tokens / total_tokens if total_tokens else 0,
        },
        {
            "reuse_type": "cold_new",
            "tokens": cold_tokens,
            "token_share": cold_tokens / total_tokens if total_tokens else 0,
        },
    ]
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], output_png: Path) -> None:
    row_map = {row["reuse_type"]: row for row in rows}
    values = [
        float(row_map["strict_position_reuse"]["token_share"]) * 100,
        float(row_map["position_shifted_pic_only"]["token_share"]) * 100,
        float(row_map["cold_new"]["token_share"]) * 100,
    ]
    labels = ["Strict position\nreuse", "Position-shifted\ncontent reuse", "Cold/new"]
    colors = ["#4E79A7", "#59A14F", "#E15759"]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.1, f"{value:.1f}%", ha="center", fontsize=11)

    content_share = float(row_map["content_reuse_total"]["token_share"]) * 100
    ax.axhline(content_share, color="0.25", linestyle="--", linewidth=1.5)
    ax.text(2.48, content_share + 1.2, f"Content reuse total {content_share:.1f}%", ha="right", fontsize=10)
    ax.set_ylabel("Token share (%)")
    ax.set_title("Previous-request context reuse breakdown")
    ax.set_ylim(0, max(max(values), content_share) * 1.22)
    ax.grid(axis="y", color="0.88")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
    write_csv(output_dir / "csv" / "context_reuse_vs_position_reuse.csv", rows)
    plot(rows, output_dir / "figures" / "context_reuse_vs_position_reuse.png")
    print(f"wrote {output_dir / 'csv' / 'context_reuse_vs_position_reuse.csv'}")
    print(f"wrote {output_dir / 'figures' / 'context_reuse_vs_position_reuse.png'}")


if __name__ == "__main__":
    main()
