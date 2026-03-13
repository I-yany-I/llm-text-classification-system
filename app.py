import gradio as gr
from src.rag_pipeline import load_data, build_vector_index, rag_classify

print("Loading RAG system...")

texts, labels = load_data()
index, model = build_vector_index(texts)

print("System ready!")


def classify_news(text):

    result = rag_classify(text, texts, labels, index, model)

    output = result[0]["generated_text"]

    return output


demo = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(lines=3, placeholder="Enter news text here..."),
    outputs="text",
    title="LLM + RAG News Classifier",
    description="News classification using RAG and LLM"
)

demo.launch()
