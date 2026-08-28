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
- **评测集**：自建 **126 题**评测集（108 道可答题、18 道拒答题；其中 28 道标注了至少两个期望文档），覆盖 it / academic / student / finance / international / refusal。
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

### 4. 运行评估

```bash
python evaluate_campus_kb.py
```

从内置评估问题集（126 题）加载问题，输出整体指标与按类别分组的指标。最新一次全量实测使用当前配置（抽取式 + 查询改写 + jieba/BM25 + FAISS + RRF + Cross-Encoder，`final_top_k=8`、`refusal_ce_threshold=-0.25`）：

| 指标 | 最新 126 题全量 | 历史 Top-5 对照 |
|------|------------|------------------------|
| `citation_hit_rate` | **100.00%** | 100.00% |
| `citation_recall_at_k` | **97.35%** | 94.70% |
| `refusal_accuracy` | **100.00%** | 100.00% |
| `false_refusal_rate` | **0.00%** | 0.00% |

finance / international 各仅 5 题，细分指标不稳定，对外只报整体。完整预测写在 `artifacts/predictions/campus_kb_eval_20260828.json`（默认不入库）。如需比较引用数量，可运行 `python evaluate_campus_kb.py --top-k 5` 或 `--top-k 8`。阈值搜索脚本：`python tune_refusal_threshold.py`。

指标来自当前本地模型、当前知识库和当前索引；修改模型、语料、分块参数或拒答阈值后，应重新构建索引并运行评测再更新对外指标。

| 指标 | 含义 |
|------|------|
| `citation_hit_rate` | 至少命中一个期望文档的问题占比 |
| `citation_recall_at_k` | 所有期望文档被检索到的比例 |
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
| `generation.backend` | `extractive`（默认）或 `llm` |
| `generation.lora_adapter_path` | LoRA adapter；为空则使用基座模型 |
| `prompt.refusal_threshold` | 关闭 Cross-Encoder 时，Top1 稠密余弦低于该值则拒答 |
| `retrieval.query_rewrite` | 检索前做校园同义词扩展（默认开启） |
| `prompt.refusal_ce_threshold` | 开启 Cross-Encoder 时，Top1 logit 低于该值则拒答（当前配置为 -0.25） |

## 目录结构（概要）

- **入口脚本：** Gradio 演示、索引构建、模型下载、离线评估。
- **配置：** 根目录 YAML，集中管理知识库与管线参数。
- **数据：** JSONL 知识库与评估问题集（条数见上文）。
- **源码：** `campus_kb_rag` 包（配置加载、分块、检索、生成、端到端流水线）。
- **评测：** 引用命中率、拒答等指标计算模块。
- **运行产物：** 本地向量索引与评测输出（默认不纳入版本控制）。

## 硬件建议

- **CPU**：可运行索引构建、检索、抽取式回答与评估。
- **GPU**：推荐用于 Qwen2 生成、Cross-Encoder 大批量重排或 LoRA adapter 推理。
- **磁盘**：预留 `models/` 本地快照（或 Hugging Face 缓存）与向量索引占用空间。

---

*技术栈：PyTorch · Transformers · PEFT · Sentence-Transformers · FAISS · rank-bm25 · Gradio*
