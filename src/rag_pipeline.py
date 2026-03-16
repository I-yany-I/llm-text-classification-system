from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import re

# -------------------------
# 清理输入文本
# -------------------------
def clean_query(text):
    return re.sub(r'^\s*\d+\.\s*', '', text)


# -------------------------
# 加载数据
# -------------------------
def load_data():

    dataset = load_dataset("ag_news")

    texts = dataset["train"]["text"][:5000]
    labels = dataset["train"]["label"][:5000]

    return texts, labels


# -------------------------
# 加载LLM
# -------------------------
def load_llm():

    model_name = "Qwen/Qwen2-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return llm


# -------------------------
# 构建向量数据库
# -------------------------
def build_vector_index(texts):

    os.makedirs("vector_store", exist_ok=True)

    index_file = "vector_store/faiss_index.index"
    embeddings_file = "vector_store/embeddings.npy"

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 如果缓存存在
    if os.path.exists(index_file) and os.path.exists(embeddings_file):

        print("Loading cached FAISS index...")

        index = faiss.read_index(index_file)

        return index, embed_model

    print("Building new vector index...")

    embeddings = embed_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    faiss.write_index(index, index_file)
    np.save(embeddings_file, embeddings)

    print("Vector index saved!")

    return index, embed_model


# -------------------------
# 相似文本检索
# -------------------------
def retrieve_similar(query, index, embed_model, k=5):

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    return indices[0]


# -------------------------
# RAG 分类
# -------------------------
def rag_classify(query, texts, labels, index, embed_model, llm):

    query = clean_query(query)

    similar_indices = retrieve_similar(query, index, embed_model)

    context = ""
    examples = []

    for i in similar_indices:

        context += f"Example: {texts[i]} -> {labels[i]}\n"

        examples.append(texts[i])

    prompt = f"""
You are an expert news classifier.

Categories:
0 = World (politics, international relations, conflicts, government)
1 = Sports (games, teams, athletes, competitions)
2 = Business (companies, finance, markets, investments, corporate expansion)
3 = Sci/Tech (technology, science, software, AI, gadgets)

Examples:
{context}

News:
{query}

Rules:
- Focus on the main topic.
- Company expansion or investments → Business
- Scientific discovery or technology → Sci/Tech

Return ONLY one number: 0, 1, 2, or 3.

Answer:
"""

    result = llm(
        prompt,
        max_new_tokens=5,
        do_sample=False,
        return_full_text=False
    )

    output = result[0]["generated_text"].strip()

    match = re.search(r'\b[0-3]\b', output)

    if match:

        label = match.group(0)

    else:

        label = "Unknown"

    return label, examples


# -------------------------
# 主函数
# -------------------------
if __name__ == "__main__":

    print("Loading dataset...")
    texts, labels = load_data()

    print("Building vector database...")
    index, embed_model = build_vector_index(texts)

    print("Loading LLM...")
    llm = load_llm()

    query = "Amazon announced plans to open three new distribution centers in Europe."

    label, examples = rag_classify(query, texts, labels, index, embed_model, llm)

    print("\nPrediction:", label)

    print("\nSimilar News:")

    for e in examples:
        print("-", e)
