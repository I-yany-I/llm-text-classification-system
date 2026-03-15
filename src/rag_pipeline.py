from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import re  # 用于清理开头数字和提取独立数字

# -------------------------
# 加载数据
# -------------------------
def load_data():
    dataset = load_dataset("ag_news")
    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]
    return texts, labels

# -------------------------
# 加载LLM (只加载一次)
# -------------------------
def load_llm():
    model_name = "Qwen/Qwen2-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return llm

# -------------------------
# 构建向量数据库（保存到 vector_store）
# -------------------------
def build_vector_index(texts):
    os.makedirs("vector_store", exist_ok=True)
    index_file = "vector_store/faiss_index.index"
    embeddings_file = "vector_store/embeddings.npy"

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    if os.path.exists(index_file) and os.path.exists(embeddings_file):
        print("Loading FAISS index and embeddings...")
        index = faiss.read_index(index_file)
        embeddings = np.load(embeddings_file)
        return index, embed_model

    print("Building new FAISS index...")
    embeddings = embed_model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, index_file)
    np.save(embeddings_file, embeddings)
    print("FAISS index and embeddings saved in vector_store/")

    return index, embed_model

# -------------------------
# 清理查询文本开头的数字序号
# -------------------------
def clean_query(text):
    # 删除开头序号或数字 + 点 + 空格
    return re.sub(r'^\s*\d+\.\s*', '', text)

# -------------------------
# 检索相似文本
# -------------------------
def retrieve_similar(query, index, embed_model, k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return indices[0]

# -------------------------
# RAG 分类
# -------------------------
def rag_classify(query, texts, labels, index, embed_model, llm):
    # 先清理开头数字
    query = clean_query(query)

    similar_indices = retrieve_similar(query, index, embed_model)
    context = ""
    for i in similar_indices:
        context += f"Example: {texts[i]} -> {labels[i]}\n"

    prompt = f"""
You are an expert news classifier.

Categories:
0 = World
1 = Sports
2 = Business
3 = Sci/Tech

Examples:
{context}

News:
{query}

Return ONLY the category number as a single digit (0,1,2,3). 
Do NOT include any text, punctuation, or numbers from the news article.

Answer:
"""

    result = llm(
        prompt,
        max_new_tokens=5,
        do_sample=False,
        return_full_text=False,
        clean_up_tokenization_spaces=True
    )

    output = result[0]["generated_text"].strip()

    # 使用正则提取独立数字 0-3
    match = re.search(r'\b[0-3]\b', output)
    if match:
        return match.group(0)
    return "Unknown"

# -------------------------
# 主函数测试
# -------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    texts, labels = load_data()

    print("Building vector database...")
    index, embed_model = build_vector_index(texts)

    print("Loading LLM...")
    llm = load_llm()

    test_queries = [
        "Massive protests erupted in the capital after the controversial election results were announced.",
        "3. Massive protests erupted in the capital after the controversial election results were announced."
    ]

    for query in test_queries:
        prediction = rag_classify(query, texts, labels, index, embed_model, llm)
        print(f"\nQuery: {query}\nPrediction: {prediction}")
