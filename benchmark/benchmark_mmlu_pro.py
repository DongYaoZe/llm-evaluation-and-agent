import json
import re
import time
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from openai import APITimeoutError, APIConnectionError, APIStatusError
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

from llm_client import LLMClient, chat_memory

# ==================== 配置 ====================

# 要评测的类别列表 (MMLU-Pro Categories)
SELECTED_CATEGORIES = [
    "computer science",
    "biology",
]

MAX_TEST_SAMPLES = 20  # 每个类别最多评测的题目数 (控制 API 消耗与速度)
NUM_WORKERS = 3        # 并发线程数 (设置为 3 以平衡速度与 API 限制)
RESULT_DIR = Path(__file__).parent / "results"

# 选项标签 (MMLU-Pro 最多有 10 个选项 A-J)
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

_print_lock = threading.Lock()  # 线程安全打印

# ==================== Prompt 模板 ====================

def build_zero_shot_prompt(question: str, choices: list[str]) -> str:
    """构造零样本 prompt"""
    parts = ["请回答以下选择题。你只能且必须输出一个选项字母（" + "/".join(CHOICE_LABELS[:len(choices)]) + "），不要输出任何其他内容，也不要解释你的推理过程。\n"]
    parts.append(question)
    for j, choice in enumerate(choices):
        parts.append(f"{CHOICE_LABELS[j]}. {choice}")
    parts.append("你的唯一答案（仅输出一个字母）:")
    return "\n".join(parts)

def extract_answer(response: str, num_options: int) -> str | None:
    """从 LLM 回复中提取 A-J 答案"""
    text = response.strip()
    valid_labels = CHOICE_LABELS[:num_options]
    char_pattern = f"[{"".join(valid_labels)}{"".join(valid_labels).lower()}]"

    # 1. 匹配 "答案是 X" 或 "答案: X" 等中文模式
    m = re.search(rf"答案[是为:：]\s*({char_pattern})", text)
    if m:
        return m.group(1).upper()

    # 2. 匹配开头的单个字母
    m = re.match(rf"^\s*({char_pattern})\s*[\.\)、，,]", text)
    if m:
        return m.group(1).upper()

    # 3. 匹配单独成行的单个字母
    m = re.search(rf"(?:^|\n)\s*({char_pattern})\s*(?:$|[\n\.\)、，])", text)
    if m:
        return m.group(1).upper()

    # 4. 直接匹配开头的单个字母
    m = re.match(rf"^\s*({char_pattern})\s*$", text)
    if m:
        return m.group(1).upper()

    # 5. 在回复中搜索最后一个出现的选项字母
    m = re.findall(rf"\b({"|".join(valid_labels)})\b", text)
    if m:
        return m[-1]

    return None

# ==================== 评测核心 ====================

def _eval_single_question(
    sample: dict,
    category: str,
    total: int,
) -> dict:
    """单题评测（供线程池调用）"""
    question = sample["question"]
    choices = sample["options"]
    correct_label = sample["answer"]  # MMLU-Pro dataset already contains "A", "B" etc. as string
    idx = sample["_idx"]
    t_start = time.time()

    with _print_lock:
        print(f"  [开始] {category} 第{idx}/{total}题 ...")

    prompt = build_zero_shot_prompt(question, choices)

    client = LLMClient()
    chat = chat_memory()
    chat.add_message("user", prompt)

    response = ""
    predicted_label = None
    num_options = len(choices)

    for attempt in range(8):
        try:
            response = client.generate_response(chat)
        except (APITimeoutError, APIConnectionError) as e:
            wait_time = min(2**attempt, 30)
            print(f"  [重试] 第{idx}题 {type(e).__name__}，{wait_time}s 后重试...")
            time.sleep(wait_time)
            continue
        except APIStatusError as e:
            wait_time = min(2**attempt, 30)
            print(f"  [重试] 第{idx}题 HTTP {e.status_code}，{wait_time}s 后重试...")
            time.sleep(wait_time)
            continue
        except Exception as e:
            wait_time = min(2**attempt, 30)
            print(f"  [重试] 第{idx}题异常 {type(e).__name__}: {e}，{wait_time}s 后重试...")
            time.sleep(wait_time)
            continue

        if not (response and response.strip()):
            print(f"  [重试] 第{idx}题空响应，重新请求...")
            continue
        predicted_label = extract_answer(response, num_options)
        if predicted_label is not None:
            break
        print(f"  [重试] 第{idx}题无法从 '{response}' 提取答案，重新请求...")
    else:
        response = response or ""
        predicted_label = None

    is_correct = predicted_label == correct_label
    elapsed = time.time() - t_start

    with _print_lock:
        status = "OK" if is_correct else "FAIL" if predicted_label is not None else "ERROR"
        print(
            f"  [{status}] {category} 第{idx}/{total}题  "
            f"预测={predicted_label or '?'}  正确={correct_label}  "
            f"耗时={elapsed:.1f}s",
            flush=True
        )

    return {
        "category": category,
        "question_id": sample["question_id"],
        "question_idx": idx,
        "question": question,
        "choices": json.dumps(choices, ensure_ascii=False),
        "correct_answer": correct_label,
        "predicted_answer": predicted_label or "?",
        "is_correct": is_correct,
        "raw_response": response,
    }

def evaluate_category(
    category: str,
    full_ds,
    max_samples: int = MAX_TEST_SAMPLES,
    num_workers: int = NUM_WORKERS,
) -> list[dict]:
    """并发评测单个类别，返回结果列表"""
    # Filter for category
    filtered_ds = [item for item in full_ds if item["category"] == category]
    
    total = min(max_samples, len(filtered_ds)) if max_samples is not None else len(filtered_ds)
    print(f"类别 '{category}' 在测试集共有 {len(filtered_ds)} 题，本次将测试前 {total} 题。")

    tasks = []
    for i in range(total):
        sample = dict(filtered_ds[i])
        sample["_idx"] = i + 1
        tasks.append(sample)

    results = []
    correct = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_map = {
            executor.submit(_eval_single_question, task, category, total): task["_idx"]
            for task in tasks
        }
        pending = set(future_map.keys())

        with tqdm(total=total, desc=f"评测 {category}", unit="题") as pbar:
            while pending:
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    results.append(result)
                    if result["is_correct"]:
                        correct += 1
                    pbar.update(1)

    results.sort(key=lambda r: r["question_idx"])
    acc = correct / total * 100 if total > 0 else 0
    print(f"  {category} 结果: {correct}/{total} = {acc:.1f}%")
    return results

def print_summary(df: pd.DataFrame):
    """汇总结果并输出"""
    total_correct = df["is_correct"].sum()
    total_count = len(df)
    acc = total_correct / total_count * 100 if total_count > 0 else 0
    print(f"\n--- MMLU-Pro 评测汇总 ---")
    print(f"总正确率: {total_correct}/{total_count} = {acc:.2f}%")

    category_stats = df.groupby("category").agg(
        总题数=("is_correct", "count"),
        正确数=("is_correct", "sum"),
        正确率=("is_correct", lambda x: f"{x.sum()/len(x)*100:.1f}%"),
    )
    print(category_stats.to_string())

def run_benchmark():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    temp_client = LLMClient()
    print(f"评测模型: {temp_client.model}")
    print(f"待评测类别: {SELECTED_CATEGORIES}")

    print("正在加载 MMLU-Pro 测试集...")
    full_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    all_results = []
    for category in SELECTED_CATEGORIES:
        print(f"\n{'='*60}")
        print(f"开始评测类别: {category}")
        results = evaluate_category(category, full_ds, max_samples=MAX_TEST_SAMPLES)
        all_results.extend(results)

        # 保存中间结果
        df = pd.DataFrame(all_results)
        df.to_csv(
            RESULT_DIR / "mmlu_pro_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

    df = pd.DataFrame(all_results)
    print_summary(df)

    # 保存最终结果和统计
    df.to_csv(RESULT_DIR / "mmlu_pro_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n所有结果已保存至: {RESULT_DIR}")

if __name__ == "__main__":
    run_benchmark()
