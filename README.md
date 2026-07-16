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

## 第三阶段个人实验

同一批 40 道 MMLU-Pro 题比较了当前直接回答、自反思、3 路自一致性和投票后反思。历史保存结果只用于观察服务端漂移，正式配对使用同一运行窗口的当前直接基线。

| 条件 | 答对/40 | 全样本准确率 | 有效回答覆盖率 | 已回答样本准确率 |
|---|---:|---:|---:|---:|
| 当前直接回答 | 33 | 82.5% | 90% | 91.67% |
| 自反思 | 28 | 70.0% | 75% | 93.33% |
| 3 路自一致性 | 31 | 77.5% | 85% | 91.18% |
| 投票后反思 | 29 | 72.5% | 80% | 90.62% |

三种缓解流程均未超过当前直接回答。主要损失来自模型在统一的 768 token 输出预算内没有返回可解析的最终字母，而不是成功返回后知识准确率大幅下降。这是负结果，不应解释为反思或自一致性在所有模型上都无效。

逐题结果、覆盖率、token、耗时和配对统计位于 benchmark/results/mitigation。

## 复现

1. python -m venv .venv
2. 激活环境并运行 pip install -r requirements.txt
3. 复制 .env.example 为 .env，填写自己的接口、模型名和密钥
4. 运行 python benchmark/benchmark_mmlu.py
5. 运行 python benchmark/benchmark_mmlu_pro.py
6. 运行第三阶段实验：python benchmark/mitigation_experiment.py --samples 3 --workers 8
7. 重新汇总覆盖率与图表：python benchmark/summarize_mitigation.py
8. ARC 原型：python arc_agent/agent.py --game ls20 --max-steps 100

第三阶段实验会进行 240 次当前 API 调用，扩大样本或重跑会产生费用。代理原型没有足够批量轨迹，仓库不宣称通关率提升。

## 安全

仓库不包含 .env、settings.json、API 密钥或个人绝对路径。历史密钥若曾以明文保存，应先在服务端撤销或轮换。

## Report

Repository URL used in the report: https://github.com/DongYaoZe/llm-evaluation-and-agent

Code license: MIT. Dataset contents follow their original licenses.
