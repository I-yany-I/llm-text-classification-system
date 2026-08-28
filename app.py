"""Gradio demo for the Nanjing University campus KB RAG system."""

from __future__ import annotations

import gradio as gr

from src.campus_kb_rag import CampusKBRAG


rag = CampusKBRAG()


def answer(question: str, top_k: int):
    result = rag.ask(question, top_k=int(top_k))
    citation_rows = [
        [
            c["index"],
            c["title"],
            c["department"],
            c["source"],
            c["updated_at"],
            round(float(c.get("score") or 0.0), 4),
        ]
        for c in result.get("citations", [])
    ]
    retrieved_text = "\n\n".join(
        [
            f"[{i}] {item['title']} | dense={float(item.get('dense_score', item.get('score', 0.0))):.4f}"
            + (
                f" | ce={float(item['cross_encoder_score']):.4f}"
                if item.get("cross_encoder_score") is not None
                else ""
            )
            + f"\n{item['text']}"
            for i, item in enumerate(result.get("retrieved", []), start=1)
        ]
    )
    diagnostic = (
        f"status={result.get('status', 'unknown')} | "
        f"refusal_reason={result.get('refusal_reason') or 'none'}"
    )
    retrieved_text = f"{diagnostic}\n\n{retrieved_text}" if retrieved_text else diagnostic
    return result["answer"], citation_rows, retrieved_text


with gr.Blocks(title="南京大学校园办事指南 RAG") as demo:
    gr.Markdown("# 南京大学校园办事指南 RAG 问答系统")
    gr.Markdown(
        "面向校园 IT / 教务办事场景：BM25 + 语义召回，BERT Cross-Encoder 重排，"
        "Prompt 约束有据作答；生成端可切换为 Qwen2 / LoRA adapter。"
    )

    with gr.Row():
        question = gr.Textbox(
            label="请输入校园办事问题",
            lines=3,
            value="校园网外怎么访问校内资源？",
        )
        top_k = gr.Slider(1, 8, value=5, step=1, label="引用片段数")

    ask_btn = gr.Button("查询")
    answer_box = gr.Textbox(label="回答", lines=7)
    citation_table = gr.Dataframe(
        headers=["引用", "标题", "部门", "来源", "更新时间", "分数"],
        label="引用来源",
    )
    retrieved_box = gr.Textbox(label="检索片段详情", lines=12)

    gr.Examples(
        examples=[
            ["校园网外怎么访问校内资源？", 5],
            ["统一身份认证密码忘记了怎么办？", 5],
            ["成绩单和在读证明应该找哪个部门？", 5],
            ["选课错过退课时间怎么办？", 5],
            ["宿舍电费怎么充值？", 5],
        ],
        inputs=[question, top_k],
    )

    ask_btn.click(answer, inputs=[question, top_k], outputs=[answer_box, citation_table, retrieved_box])


if __name__ == "__main__":
    rag.build_index(force=False)
    demo.launch()
