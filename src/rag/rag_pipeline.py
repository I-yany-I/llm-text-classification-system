from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
import os
import re

from src.llm.shared_pipeline import load_shared_generation_pipeline
from src.paths import VECTOR_STORE_DIR, ensure_project_dirs

LABEL_TEXT = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

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

    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]

    return texts, labels


# -------------------------
# 加载LLM
# -------------------------
def load_llm():
    return load_shared_generation_pipeline()


# -------------------------
# 构建向量数据库
# -------------------------
def build_vector_index(texts):
    ensure_project_dirs()

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    index_file = str(VECTOR_STORE_DIR / "faiss_index.index")
    embeddings_file = str(VECTOR_STORE_DIR / "embeddings.npy")

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
def retrieve_similar(query, index, embed_model, k=8):

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    return indices[0], distances[0]


def weighted_vote(indices, scores, labels):
    vote_scores = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    for idx, score in zip(indices, scores):
        label = int(labels[idx])
        vote_scores[label] += max(float(score), 0.0)

    best_label = max(vote_scores, key=vote_scores.get)
    return best_label, vote_scores


# -------------------------
# RAG 分类
# -------------------------
def rag_classify(query, texts, labels, index, embed_model, llm):

    query = clean_query(query)

    similar_indices, similar_scores = retrieve_similar(query, index, embed_model)
    voted_label, vote_scores = weighted_vote(similar_indices, similar_scores, labels)

    context = ""
    examples = []

    for i in similar_indices[:5]:

        lbl = int(labels[i])
        context += f"Example: {texts[i]} -> {lbl} ({LABEL_TEXT[lbl]})\n"

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
Do not output any explanation.

Answer:
"""

    result = llm(
        prompt,
        max_new_tokens=5,
        do_sample=False,
        return_full_text=False
    )

    output = result[0]["generated_text"].strip()

    match = re.search(r"[0-3]", output)

    llm_label = int(match.group(0)) if match else -1

    # Hybrid strategy:
    # 1) retrieval weighted vote gives robust baseline
    # 2) LLM can override only when retrieval is weak or ties are likely
    sorted_votes = sorted(vote_scores.values(), reverse=True)
    margin = sorted_votes[0] - sorted_votes[1]
    if llm_label in [0, 1, 2, 3] and margin < 0.15:
        final_label = llm_label
    else:
        final_label = voted_label

    return str(final_label), examples


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
