"""Summarize mitigation accuracy, answer coverage, cost, and paired changes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from mitigation_experiment import exact_mcnemar, wilson, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "results" / "mitigation" / "mitigation_item_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "results" / "mitigation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        for category in ["ALL"] + sorted({row["category"] for row in method_rows}):
            selected = method_rows if category == "ALL" else [
                row for row in method_rows if row["category"] == category
            ]
            answered = [row for row in selected if row["predicted_answer"] != "?"]
            correct_all = sum(row["is_correct"].lower() == "true" for row in selected)
            correct_answered = sum(row["is_correct"].lower() == "true" for row in answered)
            low, high = wilson(correct_all, len(selected))
            summary.append(
                {
                    "method": method,
                    "category": category,
                    "correct_all": correct_all,
                    "total": len(selected),
                    "accuracy_all_pct": round(100 * correct_all / len(selected), 2),
                    "wilson_low_pct": round(low, 2),
                    "wilson_high_pct": round(high, 2),
                    "answered": len(answered),
                    "coverage_pct": round(100 * len(answered) / len(selected), 2),
                    "accuracy_answered_pct": (
                        round(100 * correct_answered / len(answered), 2) if answered else ""
                    ),
                    "mean_latency_s": round(
                        sum(float(row["latency_s"] or 0) for row in selected) / len(selected), 3
                    ),
                    "prompt_tokens": sum(int(row["prompt_tokens"] or 0) for row in selected),
                    "completion_tokens": sum(
                        int(row["completion_tokens"] or 0) for row in selected
                    ),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mitigation_summary.csv", summary)

    baseline = {
        row["item_id"]: row
        for row in rows
        if row["method"] == "direct_current"
    }
    comparisons = []
    for method in ["self_reflection", "self_consistency_k3", "consistency_k3_reflection"]:
        treatment = {
            row["item_id"]: row
            for row in rows
            if row["method"] == method
        }
        ids = sorted(baseline, key=int)
        base_correct = [baseline[item]["is_correct"].lower() == "true" for item in ids]
        treat_correct = [treatment[item]["is_correct"].lower() == "true" for item in ids]
        fixed, broken, p_value = exact_mcnemar(base_correct, treat_correct)
        comparisons.append(
            {
                "baseline": "direct_current",
                "treatment": method,
                "wrong_to_right": fixed,
                "right_to_wrong_or_unanswered": broken,
                "net_gain": fixed - broken,
                "treatment_unanswered": sum(
                    treatment[item]["predicted_answer"] == "?" for item in ids
                ),
                "mcnemar_exact_p_failures_as_wrong": round(p_value, 6),
            }
        )
    write_csv(args.output_dir / "paired_comparisons.csv", comparisons)

    all_rows = [row for row in summary if row["category"] == "ALL"]
    order = [
        "direct_current",
        "self_reflection",
        "self_consistency_k3",
        "consistency_k3_reflection",
    ]
    all_by_method = {row["method"]: row for row in all_rows}
    labels = ["Direct", "Reflection", "SC (k=3)", "SC + Reflection"]
    accuracy = [float(all_by_method[method]["accuracy_all_pct"]) for method in order]
    coverage = [float(all_by_method[method]["coverage_pct"]) for method in order]
    x = range(len(order))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([i - 0.19 for i in x], accuracy, width=0.38, label="All-item accuracy", color="#176B87")
    ax.bar([i + 0.19 for i in x], coverage, width=0.38, label="Answer coverage", color="#D97706")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(list(x), labels)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left")
    for i, value in enumerate(accuracy):
        ax.text(i - 0.19, value + 1.2, f"{value:.1f}", ha="center", fontsize=9)
    for i, value in enumerate(coverage):
        ax.text(i + 0.19, value + 1.2, f"{value:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output_dir / "mitigation_accuracy_coverage.png", dpi=220)
    plt.close(fig)
    print(f"wrote summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
