# LLM Evaluation and ARC-AGI-3 Agent

励行导师课程个人研究报告的佐证代码，包括 MMLU/MMLU-Pro 零样本评测和 ARC-AGI-3 工具调用代理。

## 保存结果

| 基准 | 学科 | 正确/题数 | 准确率 |
|---|---|---|---|
| MMLU | computer science | 99/100 | 99.00% |
| MMLU | biology | 143/144 | 99.31% |
| MMLU-Pro | computer science | 18/20 | 90.00% |
| MMLU-Pro | biology | 17/20 | 85.00% |

MMLU-Pro 是每类前 20 题固定子集。模型名来自实验时中转服务配置，不等同于公开官方版本的权威结果。

## 复现

1. python -m venv .venv
2. 激活环境并运行 pip install -r requirements.txt
3. 复制 .env.example 为 .env，填写自己的接口、模型名和密钥
4. 运行 python benchmark/benchmark_mmlu.py
5. 运行 python benchmark/benchmark_mmlu_pro.py
6. ARC 原型：python arc_agent/agent.py --game ls20 --max-steps 100

扩大样本会产生 API 费用。代理原型没有足够批量轨迹，仓库不宣称通关率提升。

## 安全

仓库不包含 .env、settings.json、API 密钥或个人绝对路径。历史密钥若曾以明文保存，应先在服务端撤销或轮换。

## Report

Repository URL used in the report: https://github.com/DongYaoZe/llm-evaluation-and-agent

Code license: MIT. Dataset contents follow their original licenses.
