import gradio as gr
from src.rag_pipeline import load_data, build_vector_index, rag_classify, load_llm, clean_query

print("Loading RAG system...")

# -------------------------
# 加载数据
# -------------------------
texts, labels = load_data()

# -------------------------
# 构建或加载向量数据库
# -------------------------
index, embed_model = build_vector_index(texts)

# -------------------------
# 加载 LLM
# -------------------------
llm = load_llm()

print("System ready!")

# 标签映射
label_map = {
    "0": "World",
    "1": "Sports",
    "2": "Business",
    "3": "Sci/Tech"
}

# -------------------------
# 前端分类函数
# -------------------------
def classify_news(text):
    # 清理开头数字
    cleaned_text = clean_query(text)

    # 调用 RAG 分类
    label_id = rag_classify(cleaned_text, texts, labels, index, embed_model, llm)

    category = label_map.get(label_id, "Unknown")
    return f"{category} (Label {label_id})"

# -------------------------
# Gradio 界面
# -------------------------
demo = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(lines=3, placeholder="Enter news text here..."),
    outputs="text",
    title="LLM + RAG News Classifier",
    description="""
    News classification using RAG + LLM.

    Category Labels:
    0 = World
    1 = Sports
    2 = Business
    3 = Sci/Tech
    """
)

demo.launch()
