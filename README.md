# 南京大学校园办事指南 RAG 问答系统

> 面向 **南京大学学生的校园 IT / 教务办事指南**，构建一个可引用、可拒答、可消融的文本知识库问答系统。BERT、LoRA、Prompt、RAG 在一条产品链路中各司其职：文本召回、BERT 重排、Prompt 约束生成、LoRA 领域适配。

## 项目定位

本项目解决的是学生常见办事问题，例如：

- 「校园 VPN 怎么用？」
- 「统一身份认证密码忘了怎么办？」
- 「成绩单或在读证明去哪里办理？」
- 「选课、退课、重修类问题应该看哪个流程？」

系统只基于知识库片段作答，回答中返回引用来源；当知识库没有依据时明确拒答，避免把通用大模型的记忆当成学校政策。

> 内置 JSONL 知识库收录信息化服务、教务服务、学生服务、财务、出国（境）等多类办事指南，共 **75 条**文档，覆盖 VPN、统一身份认证、选课、考试、成绩、毕业论文、学位、邮箱、正版软件、校园网络、信息安全、在读证明、成绩单等高频场景。

### 数据说明

- **内容来源**：知识库内容基于南京大学各主管部门（信息化建设管理服务中心、教务处、学生服务中心等）的公开信息整理而成，非官方数据库导出。文本经过结构化改写和元数据标注，以适配 RAG 系统的检索和引用需求。
- **准确性**：内容力求反映公开信息，但可能已过时或不完整。正式部署时应以各主管部门最新官方通知为准，本系统仅作为技术演示。
- **评测集**：自建 **126 题**评测集（106 道可答题、20 道拒答题；其中 28 道标注了至少两个期望文档，用于多文档引用），覆盖 it / academic / student / finance / international / refusal 六大类别。
- **编排说明**：本项目是 **Python 流水线 RAG**（BM25 + FAISS 稠密召回 + RRF + Cross-Encoder + 引用/拒答），**不是** LangGraph Agent。不把「状态图编排」写成已实现能力。

## 系统架构

```
用户问题
  │
  ▼
BM25 关键词召回 + Sentence-Transformer 稠密召回（FAISS）
  │
  ▼
RRF 融合候选
  │
  ▼
BERT Cross-Encoder 重排（可消融对比）
  │
  ▼
Top-N 引用片段
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
| 稀疏检索 | BM25（rank-bm25）+ RRF 融合 | 保留「VPN」「统一身份认证」「成绩单」等关键词优势 |
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

### 2. 构建校园知识库索引

```bash
python build_campus_kb_index.py
```

默认读取配置中的知识库数据，将 FAISS 索引与元数据写入本地索引目录（首次运行自动生成）。强制重建索引：

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

### 4. 运行评估

```bash
python evaluate_campus_kb.py
```

从内置评估问题集（126 题）加载问题，输出整体指标与按类别分组的指标。对外引用的 85.5% / 82.1% / 80.0% 来自早期 **81 题切片**，126 题全量需重新跑评测后才能替换。

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
| `retrieval.hybrid_enabled` | 是否启用 BM25 + 稠密召回融合 |
| `retrieval.cross_encoder.enabled` | 是否启用 BERT Cross-Encoder 重排 |
| `generation.backend` | `extractive`（默认）或 `llm` |
| `generation.lora_adapter_path` | LoRA adapter；为空则使用基座模型 |
| `prompt.refusal_threshold` | 低相似度时触发拒答 |

## 目录结构（概要）

- **入口脚本：** Gradio 演示、索引构建、离线评估。
- **配置：** 根目录 YAML，集中管理知识库与管线参数。
- **数据：** JSONL 知识库与评估问题集（条数见上文）。
- **源码：** `campus_kb_rag` 包（配置加载、分块、检索、生成、端到端流水线）。
- **评测：** 引用命中率、拒答等指标计算模块。
- **运行产物：** 本地向量索引与评测输出（默认不纳入版本控制）。

## 硬件建议

- **CPU**：可运行索引构建、检索、抽取式回答与评估。
- **GPU**：推荐用于 Qwen2 生成、Cross-Encoder 大批量重排或 LoRA adapter 推理。
- **磁盘**：预留 Hugging Face 模型缓存与本地向量索引占用空间。

---

*技术栈：PyTorch · Transformers · PEFT · Sentence-Transformers · FAISS · rank-bm25 · Gradio*
