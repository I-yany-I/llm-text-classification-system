# Evaluation Results

本页记录当前对外引用指标的评测口径。指标用于验证本地演示系统，不代表南京大学官方服务效果，也不能外推到真实线上流量。

## 数据划分

- 知识库：77 条结构化校园办事语料。
- 完整评测集：180 题，其中 155 题可回答、25 题应拒答。
- dev：52 题，只用于阈值与配置选择。
- test：128 题，只用于最终报告，其中 110 题可回答、18 题应拒答。
- 数据文件：`data/campus_kb/eval_questions_dev.jsonl`、`data/campus_kb/eval_questions_test.jsonl`。

## Test 结果

运行配置：抽取式回答、确定性查询改写、jieba/BM25 + Sentence-Transformer/FAISS、RRF、Cross-Encoder、`final_top_k=8`、`refusal_ce_threshold=-0.25`。

| 指标 | 结果 |
|---|---:|
| Citation Hit Rate | 100.00% |
| Citation Recall@8 | 98.26% |
| Citation MRR | 95.91% |
| Refusal Accuracy | 100.00% |
| False Refusal Rate | 0.00% |

指标定义：

- Citation Hit Rate：可回答问题中，至少命中一个期望文档的问题比例。
- Citation Recall@8：全部期望文档中进入前 8 条引用的比例。
- Citation MRR：每道可回答问题首个期望文档倒数排名的平均值。
- Refusal Accuracy：应拒答问题中被正确拒答的比例。
- False Refusal Rate：可回答问题中被错误拒答的比例。

## 消融结果

所有方案使用同一 test 128 题。

| 方案 | Citation Recall@8 | Citation MRR | Refusal Accuracy |
|---|---:|---:|---:|
| dense only | 93.02% | 83.21% | 83.33% |
| hybrid | 98.84% | 92.08% | 83.33% |
| hybrid + rewrite | 97.67% | 91.74% | 83.33% |
| full | 98.26% | 95.91% | 100.00% |

完整链路相对 dense only 的 MRR 提升 12.70 个百分点。混合召回主要提高覆盖率，Cross-Encoder 与拒答门控进一步改善首个正确证据排名和边界问题处理。

## 复现

```bash
python build_campus_kb_index.py --force
python evaluate_campus_kb.py --split dev --top-k 8 --output artifacts/predictions/dev.json
python evaluate_campus_kb.py --split test --top-k 8 --output artifacts/predictions/test.json
python evaluate_ablation.py --split test --output artifacts/predictions/ablation.json
```

也可以执行：

```bash
python reproduce_eval.py
```

评测结果依赖当前语料、模型快照、分块参数和阈值。任一项发生变化后必须重建索引并重新评测，不能继续沿用本页数字。

## 已知限制

- 评测集为自建离线数据，与知识库来自同一校园场景，不能替代真实用户测试。
- finance 和 international 子类样本仍少，不单独对外报告类别指标。
- 部分知识条目来自官方栏目入口或结构化摘要，并非全部页面都能公开抓取正文。
- 当前默认回答为抽取式，主要评估检索、引用和拒答，不代表开放式生成质量。
