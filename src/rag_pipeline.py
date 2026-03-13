from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np
from transformers import pipeline

# 加载数据
def load_data():

    dataset = load_dataset("ag_news")

    texts = dataset["train"]["text"][:5000]
    labels = dataset["train"]["label"][:5000]

    return texts, labels

# 加载LLM
def load_llm():

    llm = pipeline(
        "text-generation",
        model="gpt2"
    )

    return llm

# 建立向量库
def build_vector_index(texts):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(texts)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings))

    print("Vector index built!")

    return index, model

# 返回相似的前k个向量序号
def retrieve_similar(query, index, model, k=3):

    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, k)

    return indices[0]

# 构造RAG prompt
def rag_classify(query, texts, labels, index, model):

    llm = load_llm()

    similar_indices = retrieve_similar(query, index, model)

    context = ""

    for i in similar_indices:
        context += f"Example: {texts[i]} -> Label{labels[i]}\n"

    prompt = f"""
You are a news classifier.

Categories:
0 World
1 Sports
2 Business
3 Sci/Tech

Here are similar news examples:

{context}

Now classify this news:

{query}

Answer with ONLY the category number.
Category:
"""

    result = llm(
        prompt,
        max_new_tokens=2,
        do_sample=False
    )

    return result

if __name__ == "__main__":

    texts, labels = load_data()

    index, model = build_vector_index(texts)

    query = "Apple releases new AI chip"

    result = rag_classify(query, texts, labels, index, model)

    print(result)