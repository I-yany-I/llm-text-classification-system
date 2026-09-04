# Campus RAG Interview Handbook Implementation Plan

> **For agentic workers:** This document records the approved documentation task. It is a writing task, so validation focuses on source-code traceability, resume consistency, and rendered document usability rather than production-code TDD.

**Goal:** Create an autumn-recruitment interview handbook for the NJU campus RAG project and update the resume project entry to match the current implementation and verified evaluation protocol.

**Architecture:** The handbook will mirror the existing project-handbook style: project positioning, end-to-end architecture, code map, resume-bullet stories, development retrospectives, metrics, interview Q&A, and preparation checklist. It will cite concrete modules and distinguish implemented behavior, verified offline metrics, and future work.

**Tech Stack:** Markdown, LaTeX resume source, Python RAG pipeline, YAML configuration, FAISS, Sentence-Transformers, BM25/jieba, Cross-Encoder, Gradio, pytest.

---

### Task 1: Map the project and resume source of truth

**Files:**
- Read: `个人项目/nju-campus-kb-rag/README.md`
- Read: `个人项目/nju-campus-kb-rag/INTERVIEW_PREP.md`
- Read: `个人项目/nju-campus-kb-rag/src/campus_kb_rag/*.py`
- Read: `resume_zh.tex`

- [x] Confirm the current pipeline, configuration, evaluation split, verified metrics, and known limitations.
- [x] Confirm the resume currently contains the stale 126-question metrics and identify the exact project bullets to replace.

### Task 2: Write the interview handbook

**Files:**
- Create: `项目介绍/南京大学校园办事指南RAG_项目掌握与秋招面试手册.md`

- [x] Include a one-sentence project definition and a text architecture diagram.
- [x] Explain each implementation boundary with concrete file/function references.
- [x] Decompose every resume bullet into multiple interview stories with follow-up questions.
- [x] Record the iterative problems and the engineering decisions that resolved them.
- [x] Include honest metric, threshold, reproducibility, limitation, and future-work language.
- [x] End with concise 30-second, 2-minute, and technical versions plus a preparation checklist.

### Task 3: Synchronize the resume

**Files:**
- Modify: `resume_zh.tex`

- [x] Replace outdated 126-question metrics with the current 180-question dev/test protocol.
- [x] Keep the resume concise and avoid claiming online traffic, production deployment, or LangGraph orchestration.
- [x] Preserve the existing GitHub link and the project’s three-bullet structure.

### Task 4: Validate the deliverables

**Files:**
- Verify: handbook, `resume_zh.tex`, project tests, generated resume PDF if LaTeX is available.

- [x] Search for stale campus-RAG claims in the resume and handbook.
- [x] Run the complete project test suite.
- [x] Compile the resume and confirm the PDF is generated without LaTeX errors.
