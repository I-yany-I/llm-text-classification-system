# 南京大学校园办事指南 RAG 问答系统

> 面向 **南京大学学生的校园 IT / 教务办事指南**，构建一个可引用、可拒答、可消融的文本知识库问答系统。BERT、LoRA、Prompt、RAG 在一条产品链路中各司其职：文本召回、BERT 重排、Prompt 约束生成、LoRA 领域适配。

## 项目定位

本项目解决的是学生常见办事问题，例如：

- 「校园 VPN 怎么用？」
- 「统一身份认证密码忘了怎么办？」
- 「成绩单或在读证明去哪里办理？」
- 「选课、退课、重修类问题应该看哪个流程？」

系统只基于知识库片段作答，回答中返回引用来源；当知识库没有依据时明确拒答，避免把通用大模型的记忆当成学校政策。

> 内置 JSONL 知识库收录信息化服务、教务服务、学生服务、财务、出国（境）等多类办事指南，共 **77 条**文档，覆盖 VPN、统一身份认证、选课、考试、成绩、毕业论文、学位、邮箱、正版软件、校园网络、信息安全、在读证明、成绩单等高频场景。

### 数据说明

- **内容来源**：条目先按主管部门公开信息结构化整理，再把能抓到正文的官网页写回对应文档。已经核对过正文的页面包括本科生院 [2026 年春季开学教务事项](https://jw.nju.edu.cn/83/24/c26263a820004/page.htm)、[出国成绩/学历证明](https://jw.nju.edu.cn/a9/32/c24739a370994/page.htm)、[学生证补办](https://jw.nju.edu.cn/24751/list.htm)、[教服平台](https://jw.nju.edu.cn/24777/list.htm)，以及信息化中心 [自助打印](https://itsc.nju.edu.cn/21426/list.htm)、[校外 VPN](https://itsc.nju.edu.cn/21601/list.htm)、[统一身份认证](https://itsc.nju.edu.cn/tysfrz/list.htm)、[正版软件](https://itsc.nju.edu.cn/zbrj/mainm.htm)、[校园卡补卡](https://itsc.nju.edu.cn/21446/list.htm)。选课平台、ehall 等登录后系统没有抓。部分栏目页只有导航壳，正文仍用结构化摘要并保留官方入口。
- **准确性**：内容力求反映公开信息，但可能已过时或不完整。正式办事请以主管部门最新通知为准，本系统仅作为技术演示。
- **评测集**：自建 **180 题**评测集（155 道可答题、25 道拒答题；其中 64 道标注了至少两个期望文档），覆盖 it / academic / student / finance / international / refusal。
- **编排说明**：本项目是 **Python 流水线 RAG**（查询改写 + jieba/BM25 + FAISS + RRF + Cross-Encoder + 引用/拒答），**不是** LangGraph Agent。

## 系统架构

```
用户问题
  │
  ▼
确定性查询改写（校园同义词，不是 LLM）
  │
  ▼
BM25（jieba + 单字/bigram）+ Sentence-Transformer / FAISS
  │
  ▼
RRF 融合候选
  │
  ▼
BERT Cross-Encoder 重排（可消融对比）
  │
  ▼
验证集搜索得到的 CE 拒答阈值
  │
  ▼
抽取式回答 / 可选 Qwen2、LoRA
  │
  ▼
带引用的中文回答 + 拒答判断
```

## 技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 知识库 | JSONL 文档 + 段落分块 | 含 `source`、`source_type`、`updated_at`、`collected_at` 等元数据 |
| 稠密检索 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + FAISS | 适合中文问句与办事片段的语义召回 |
| 稀疏检索 | BM25（jieba + 中文单字/bigram）+ RRF | 词边界与关键词（VPN、成绩单）一起保留 |
| BERT 重排 | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 对 `(问题, 片段)` 精排，提高引用命中率 |
| Prompt | 有据作答、引用格式、拒答策略 | 约束模型只依据检索证据回答 |
| LoRA | PEFT adapter 可选加载 | 用少量问答对适配校园办事口吻与输出格式 |
| 生成端 | `Qwen/Qwen2-1.5B-Instruct` 或抽取式 fallback | 默认抽取式回答，配置 `generation.backend: llm` 开启 LLM |
| 演示 | Gradio | 单轮问答、引用片段、来源与检索分数展示 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

建议使用 Python 3.10+。CPU 可运行检索和抽取式回答；开启 Qwen2 生成时建议使用 CUDA。

导入应用或构造 RAG 对象不会加载模型权重；只有构建索引或第一次执行稠密检索时才会加载 embedding，启用 Cross-Encoder 后第一次重排时才加载重排模型。默认抽取式回答也不会加载 Qwen2。国内网络下 Hugging Face 可能卡住，可先把权重下到项目内 `models/`（已 gitignore）：

```bash
python download_models.py
```

脚本默认走 ModelScope HTTP，失败再回退 Hugging Face；之后流水线会优先用本地快照，不必改配置里的模型 ID。

### 2. 构建校园知识库索引

```bash
python build_campus_kb_index.py
```

默认读取配置中的知识库数据，将 FAISS 索引、chunk 元数据和 `manifest.json` 写入本地索引目录。manifest 会记录语料 SHA-256、embedding 模型、分块参数、向量维度和数量；语料、模型或分块配置变化后，查询不会静默复用旧索引，需显式强制重建：

```bash
python build_campus_kb_index.py --force
```

知识库 JSONL 在建索引前会校验必填字段、`tags` 类型和重复文档 ID；错误会报告具体文件和行号。配置中的分块、召回数量、RRF 和生成后端参数也会在启动时校验，非法值会直接报错而不会等到模型加载后才失败。重建过程先写入临时目录并校验完整 artifact，发布失败时会保留上一份索引。

扩充知识库只需在 JSONL 中追加条目，字段说明：

```json
{
  "id": "nju-it-vpn",
  "title": "校园 VPN 使用说明",
  "department": "信息化建设管理服务中心",
  "source": "https://istic.nju.edu.cn/vpn",
  "source_type": "official_page",
  "updated_at": "2026-05-06",
  "collected_at": "2026-05-06",
  "tags": ["VPN", "校外访问"],
  "text": "正文内容..."
}
```

### 3. 启动 Gradio 演示

```bash
python app.py
```

页面展示回答、引用片段、来源标题和检索分数。示例问题：

- `校园网外怎么访问校内资源？`
- `统一身份认证密码忘记了怎么办？`
- `成绩单和在读证明应该找哪个部门？`

`CampusKBRAG.ask()` 返回结构化结果：`status` 为 `answered`、`refused` 或 `input_required`；当拒答时，`refusal_reason` 会标记为 `out_of_scope`、`low_confidence` 或 `sentinel_document`。`search_query` 保存实际用于检索的标准化/改写问题，便于排查召回效果。

启动前可运行轻量健康检查：

```bash
python health.py
```

健康检查会验证配置、知识库文件、FAISS/metadata/manifest 契约和本地模型来源；索引校验只读取已有 artifact，不会加载 embedding、Cross-Encoder 或生成模型。所有必需检查通过时退出码为 `0`，否则为 `1`，并输出结构化 JSON 诊断。

### 4. 运行评估

```bash
python evaluate_campus_kb.py --split dev
python evaluate_campus_kb.py --split test
```

内置评估问题集拆成 `dev` / `test` 两份：`dev` 52 题用于调参，`test` 128 题用于对外报告；`full` 180 题仍可通过 `--split full` 读取。阈值搜索默认读取完整集，用固定 hash 桶选择 `dev`，只在 `dev` 上选阈值并同时输出 `test` 观察值：`python tune_refusal_threshold.py`。这样不会把测试集用于阈值选择。

| 切分 | 题数 | `citation_hit_rate` | `citation_recall_at_k` | `citation_mrr` | `refusal_accuracy` | `false_refusal_rate` |
|------|------:|--------------------:|-----------------------:|---------------:|--------------------:|----------------------:|
| `dev` | 52 | **100.00%** | **95.24%** | **93.78%** | **100.00%** | **0.00%** |
| `test` | 128 | **100.00%** | **98.26%** | **95.91%** | **100.00%** | **0.00%** |

评测口径、消融结果、复现命令和适用边界见 [`docs/evaluation-results.md`](docs/evaluation-results.md)。

finance / international 现在分别有 14 / 13 题，细分指标比之前稳定一些，但仍建议对外只报整体。完整预测分别写在 `artifacts/predictions/campus_kb_eval_dev_phase7_20260830.json` 和 `artifacts/predictions/campus_kb_eval_test_phase7_20260830.json`（默认不入库）。如需比较引用数量，可运行 `python evaluate_campus_kb.py --split test --top-k 5` 或 `--split test --top-k 8`。若要看消融结果，可运行 `python evaluate_ablation.py --split test`，对应产物为 `artifacts/predictions/campus_kb_ablation_test_phase7_20260830.json`。

如需把 dev/test、消融和阈值搜索结果汇总成一份可读报告，可运行：

```bash
python report_eval.py --dev artifacts/predictions/campus_kb_eval_dev_phase7_20260830.json --test artifacts/predictions/campus_kb_eval_test_phase7_20260830.json --ablation artifacts/predictions/campus_kb_ablation_test_phase7_20260830.json --threshold artifacts/predictions/threshold_search.json
```

如果要一口气跑完建索引、dev/test 评测、消融、阈值搜索和报告，也可以直接：

```bash
python reproduce_eval.py
```

指标来自当前本地模型、当前知识库和当前索引；修改模型、语料、分块参数或拒答阈值后，应重新构建索引并运行评测再更新对外指标。阈值搜索会回放线上同一套拒答规则，包括边界文档过滤、CE 阈值和 dense fallback；最终是否采用新阈值仍需人工检查 dev/test 结果后再改主配置。

### 消融对比

在 `test` 切分上，检索链路的消融结果如下：

| 方案 | `citation_recall_at_8` | `citation_mrr` | `refusal_accuracy` |
|------|-----------------------:|---------------:|-------------------:|
| `dense_only` | **93.02%** | **83.21%** | **83.33%** |
| `hybrid` | **98.84%** | **92.08%** | **83.33%** |
| `hybrid_rewrite` | **97.67%** | **91.74%** | **83.33%** |
| `full` | **98.26%** | **95.91%** | **100.00%** |

这组结果说明：BM25 + 向量混合召回相较仅向量检索明显提升覆盖率；当前规则式查询改写在该 test 集上没有带来稳定增益，仍需扩充规则并在 dev 集调优；完整链路加入 Cross-Encoder 与拒答门控后，首个正确证据排名和边界问题处理达到最佳结果。

| 指标 | 含义 |
|------|------|
| `citation_hit_rate` | 至少命中一个期望文档的问题占比 |
| `citation_recall_at_k` | 所有期望文档被检索到的比例 |
| `citation_mrr` | 首个期望文档倒数排名的平均值 |
| `citation_recall_at_1/3/5/8` | 截断在对应引用数量下的文档召回率 |
| `refusal_accuracy` | 知识库外问题被正确拒答的比例 |
| `false_refusal_rate` | 有答案但系统拒答的比例（越低越好） |

## 配置说明

主配置文件位于仓库根目录，YAML 格式，可调知识库来源、检索与重排开关、生成后端与拒答阈值等。常用项含义如下：

| 配置项 | 说明 |
|--------|------|
| `knowledge_base.path` | 知识库数据文件位置 |
| `index.manifest_path` | 索引契约文件；记录模型、语料和向量元数据，加载前会校验 |
| `retrieval.hybrid_enabled` | 是否启用 BM25 + 稠密召回融合 |
| `retrieval.cross_encoder.enabled` | 是否启用 BERT Cross-Encoder 重排 |
| `retrieval.max_chunks_per_doc` | 单个文档最多进入最终引用结果的 chunk 数，默认 2 |
| `generation.backend` | `extractive`（默认）或 `llm` |
| `generation.lora_adapter_path` | LoRA adapter；为空则使用基座模型 |
| `prompt.refusal_threshold` | 关闭 Cross-Encoder 时，Top1 稠密余弦低于该值则拒答 |
| `retrieval.query_rewrite` | 检索前做校园同义词扩展（默认开启） |
| `prompt.refusal_ce_threshold` | 开启 Cross-Encoder 时，Top1 logit 达不到该值则进入 dense 兜底判断（当前配置为 -0.25） |
| `prompt.refusal_dense_fallback_threshold` | CE 保守时的 dense 相似度放行阈值（当前配置为 0.43） |

配置加载会拒绝空路径、非正整数检索参数、非法分块重叠范围、未知生成后端，以及启用 Cross-Encoder 但未提供模型名的配置。索引构建失败不会静默删除已有的 FAISS、chunk 元数据或 manifest 文件。

## 目录结构（概要）

- **入口脚本：** Gradio 演示、索引构建、模型下载、离线评估。
- **配置：** 根目录 YAML，集中管理知识库与管线参数。
- **数据：** JSONL 知识库与评估问题集（条数见上文）。
- **源码：** `campus_kb_rag` 包（配置加载、分块、检索、生成、端到端流水线）。
- **评测：** 引用命中率、MRR、Recall@k、拒答等指标计算模块。
- **诊断：** `health.py` 轻量启动与索引契约检查。
- **运行产物：** 本地向量索引与评测输出（默认不纳入版本控制）。

## 硬件建议

- **CPU**：可运行索引构建、检索、抽取式回答与评估。
- **GPU**：推荐用于 Qwen2 生成、Cross-Encoder 大批量重排或 LoRA adapter 推理。
- **磁盘**：预留 `models/` 本地快照（或 Hugging Face 缓存）与向量索引占用空间。

---

*技术栈：PyTorch · Transformers · PEFT · Sentence-Transformers · FAISS · rank-bm25 · Gradio*
