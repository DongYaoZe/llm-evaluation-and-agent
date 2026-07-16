"""Run a paired hallucination-mitigation experiment on saved MMLU-Pro items."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv
from openai import OpenAI

LABELS = tuple("ABCDEFGHIJ")
WRITE_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=here / "results" / "mmlu_pro_results.csv")
    parser.add_argument("--output-dir", type=Path, default=here / "results" / "mitigation")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def load_config(env_file: Path | None) -> tuple[str, str, str]:
    if env_file:
        dotenv.load_dotenv(env_file)
    else:
        dotenv.load_dotenv()
    key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    model = os.getenv("MODEL")
    if not all((key, base_url, model)):
        raise RuntimeError("API_KEY, BASE_URL and MODEL must be configured")
    return key, base_url, model


def parse_choices(raw: str) -> list[str]:
    choices = json.loads(raw)
    if not isinstance(choices, list):
        raise ValueError("choices must be a JSON list")
    return [str(choice) for choice in choices]


def question_text(row: dict[str, str]) -> str:
    choices = parse_choices(row["choices"])
    rendered = "\n".join(f"{LABELS[i]}. {choice}" for i, choice in enumerate(choices))
    return f"{row['question']}\n{rendered}"


def extract_final(text: str, valid: tuple[str, ...]) -> str | None:
    valid_pattern = "".join(valid)
    patterns = [
        rf"FINAL\s*[:：]\s*([{valid_pattern}])",
        rf"最终答案\s*[是为:：]\s*([{valid_pattern}])",
        rf"答案\s*[是为:：]\s*([{valid_pattern}])",
        rf"(?:^|\n)\s*([{valid_pattern}])\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.upper())
        if matches:
            return matches[-1]
    return None


class Caller:
    def __init__(self, key: str, base_url: str, model: str):
        self.key = key
        self.base_url = base_url
        self.model = model

    def call(self, prompt: str, temperature: float) -> dict:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                client = OpenAI(api_key=self.key, base_url=self.base_url, timeout=180)
                started = time.perf_counter()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=768,
                    stream=False,
                )
                elapsed = time.perf_counter() - started
                text = response.choices[0].message.content or ""
                usage = response.usage
                return {
                    "text": text,
                    "latency_s": round(elapsed, 3),
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                }
            except Exception as exc:
                last_error = exc
                time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"API call failed after retries: {type(last_error).__name__}") from last_error


def reflect_prompt(row: dict[str, str], initial: str) -> str:
    valid = LABELS[: len(parse_choices(row["choices"]))]
    return (
        "请独立复核下面的单项选择题。初始回答可能正确，也可能错误。"
        "先检查题意、每个关键选项和初始回答，不要为了修改而修改。"
        f"最后必须单独一行输出 FINAL: X，其中X只能是{'/'.join(valid)}。\n\n"
        f"{question_text(row)}\n\n初始回答：{initial}"
    )


def sample_prompt(row: dict[str, str]) -> str:
    valid = LABELS[: len(parse_choices(row["choices"]))]
    return (
        "请独立推理并回答下面的单项选择题。请先简要分析，再在最后单独一行输出"
        f" FINAL: X，其中X只能是{'/'.join(valid)}。不要参考其他候选答案。\n\n"
        f"{question_text(row)}"
    )


def direct_prompt(row: dict[str, str]) -> str:
    valid = LABELS[: len(parse_choices(row["choices"]))]
    return (
        "请回答下面的单项选择题。不要解释，只在最后单独一行输出"
        f" FINAL: X，其中X只能是{'/'.join(valid)}。\n\n{question_text(row)}"
    )


def run_parallel(tasks: list[tuple], fn, workers: int) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, *task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def wilson(correct: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return 100 * (center - half), 100 * (center + half)


def exact_mcnemar(baseline: list[bool], treatment: list[bool]) -> tuple[int, int, float]:
    fixed = sum((not a) and b for a, b in zip(baseline, treatment))
    broken = sum(a and (not b) for a, b in zip(baseline, treatment))
    discordant = fixed + broken
    if discordant == 0:
        return fixed, broken, 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(discordant, i) for i in range(k + 1)) / (2**discordant)
    return fixed, broken, min(1.0, 2 * tail)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    key, base_url, model = load_config(args.env_file)
    caller = Caller(key, base_url, model)
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))

    for index, row in enumerate(source):
        row["_id"] = str(index)
        row["_valid"] = "".join(LABELS[: len(parse_choices(row["choices"]))])

    historical = []
    for row in source:
        prediction = row["predicted_answer"]
        historical.append(
            {
                "item_id": row["_id"],
                "category": row["category"],
                "question_id": row["question_id"],
                "question_idx": row["question_idx"],
                "method": "direct_saved_historical",
                "correct_answer": row["correct_answer"],
                "initial_answer": prediction,
                "predicted_answer": prediction,
                "is_correct": prediction == row["correct_answer"],
                "latency_s": "",
                "prompt_tokens": "",
                "completion_tokens": "",
                "vote_counts": "",
                "raw_response": row["raw_response"],
            }
        )

    def direct_one(row: dict[str, str]) -> dict:
        result = caller.call(direct_prompt(row), temperature=0)
        answer = extract_final(result["text"], tuple(row["_valid"])) or "?"
        return {
            "item_id": row["_id"],
            "category": row["category"],
            "question_id": row["question_id"],
            "question_idx": row["question_idx"],
            "method": "direct_current",
            "correct_answer": row["correct_answer"],
            "initial_answer": "",
            "predicted_answer": answer,
            "is_correct": answer == row["correct_answer"],
            "latency_s": result["latency_s"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "vote_counts": "",
            "raw_response": result["text"],
        }

    current = run_parallel([(row,) for row in source], direct_one, args.workers)
    current_by_id = {r["item_id"]: r["predicted_answer"] for r in current}

    def reflect_one(row: dict[str, str], initial: str, method: str) -> dict:
        valid = tuple(row["_valid"])
        result = caller.call(reflect_prompt(row, initial), temperature=0)
        answer = extract_final(result["text"], valid) or "?"
        return {
            "item_id": row["_id"],
            "category": row["category"],
            "question_id": row["question_id"],
            "question_idx": row["question_idx"],
            "method": method,
            "correct_answer": row["correct_answer"],
            "initial_answer": initial,
            "predicted_answer": answer,
            "is_correct": answer == row["correct_answer"],
            "latency_s": result["latency_s"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "vote_counts": "",
            "raw_response": result["text"],
        }

    reflection = run_parallel(
        [(row, current_by_id[row["_id"]], "self_reflection") for row in source],
        reflect_one,
        args.workers,
    )

    def sample_one(row: dict[str, str], sample_index: int) -> dict:
        result = caller.call(sample_prompt(row), temperature=0.7)
        answer = extract_final(result["text"], tuple(row["_valid"])) or "?"
        return {"item_id": row["_id"], "sample": sample_index, "answer": answer, **result}

    samples = run_parallel(
        [(row, sample_index) for row in source for sample_index in range(args.samples)],
        sample_one,
        args.workers,
    )
    by_item: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_item[sample["item_id"]].append(sample)

    consistency = []
    majority_by_id = {}
    for row in source:
        item_samples = sorted(by_item[row["_id"]], key=lambda x: x["sample"])
        valid_answers = [x["answer"] for x in item_samples if x["answer"] != "?"]
        counts = Counter(valid_answers)
        majority = counts.most_common(1)[0][0] if counts else "?"
        majority_by_id[row["_id"]] = majority
        consistency.append(
            {
                "item_id": row["_id"],
                "category": row["category"],
                "question_id": row["question_id"],
                "question_idx": row["question_idx"],
                "method": f"self_consistency_k{args.samples}",
                "correct_answer": row["correct_answer"],
                "initial_answer": "",
                "predicted_answer": majority,
                "is_correct": majority == row["correct_answer"],
                "latency_s": round(sum(x["latency_s"] for x in item_samples), 3),
                "prompt_tokens": sum((x["prompt_tokens"] or 0) for x in item_samples),
                "completion_tokens": sum((x["completion_tokens"] or 0) for x in item_samples),
                "vote_counts": json.dumps(counts, ensure_ascii=False, sort_keys=True),
                "raw_response": json.dumps([x["text"] for x in item_samples], ensure_ascii=False),
            }
        )

    hybrid = run_parallel(
        [(row, majority_by_id[row["_id"]], f"consistency_k{args.samples}_reflection") for row in source],
        reflect_one,
        args.workers,
    )
    all_rows = historical + current + reflection + consistency + hybrid
    all_rows.sort(key=lambda r: (r["method"], int(r["item_id"])))
    write_csv(args.output_dir / "mitigation_item_results.csv", all_rows)

    baseline_by_id = {r["item_id"]: bool(r["is_correct"]) for r in current}
    summary = []
    for method in sorted({r["method"] for r in all_rows}):
        method_rows = [r for r in all_rows if r["method"] == method]
        for category in ["ALL"] + sorted({r["category"] for r in method_rows}):
            selected = method_rows if category == "ALL" else [r for r in method_rows if r["category"] == category]
            correct = sum(bool(r["is_correct"]) for r in selected)
            low, high = wilson(correct, len(selected))
            latencies = [float(r["latency_s"]) for r in selected if r["latency_s"] != ""]
            summary.append(
                {
                    "method": method,
                    "category": category,
                    "correct": correct,
                    "total": len(selected),
                    "accuracy_pct": round(100 * correct / len(selected), 2),
                    "wilson_low_pct": round(low, 2),
                    "wilson_high_pct": round(high, 2),
                    "mean_latency_s": round(statistics.mean(latencies), 3) if latencies else "",
                    "prompt_tokens": sum(int(r["prompt_tokens"] or 0) for r in selected),
                    "completion_tokens": sum(int(r["completion_tokens"] or 0) for r in selected),
                }
            )
    write_csv(args.output_dir / "mitigation_summary.csv", summary)

    comparisons = []
    baseline_vector = [baseline_by_id[str(i)] for i in range(len(source))]
    excluded = {"direct_current", "direct_saved_historical"}
    for method in sorted({r["method"] for r in all_rows} - excluded):
        rows = sorted((r for r in all_rows if r["method"] == method), key=lambda r: int(r["item_id"]))
        vector = [bool(r["is_correct"]) for r in rows]
        fixed, broken, p_value = exact_mcnemar(baseline_vector, vector)
        comparisons.append(
            {
                "baseline": "direct_current",
                "treatment": method,
                "wrong_to_right": fixed,
                "right_to_wrong": broken,
                "net_gain": fixed - broken,
                "mcnemar_exact_p": round(p_value, 6),
            }
        )
    write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    metadata = {
        "model_service_identifier": model,
        "input": str(args.input.name),
        "items": len(source),
        "self_consistency_samples": args.samples,
        "max_output_tokens": 768,
        "workers": args.workers,
        "note": "Formal paired comparisons use the current direct baseline. Saved direct results are historical only. The model name is a service-side identifier and is not treated as an official public model version.",
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
