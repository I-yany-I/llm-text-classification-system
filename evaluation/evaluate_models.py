from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import torch
from transformers import AutoTokenizer


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ===== 导入模型 =====
from src.models.bert_model import load_model
from src.models.lora_model import load_lora_model
from src.prompt.prompt_classifier import load_llm
from src.rag.rag_pipeline import (
    load_data,
    build_vector_index,
    rag_classify,
)

# =============================
# 1 加载测试数据
# =============================

print("Loading dataset...")

dataset = load_dataset("ag_news")

test_texts = dataset["test"]["text"][:200]
test_labels = dataset["test"]["label"][:200]

print("Test samples:", len(test_texts))


# =============================
# 2 通用评测函数
# =============================

def evaluate_model(name, predict_function):

    predictions = []

    for text in tqdm(test_texts, desc=f"Evaluating {name}"):

        pred = predict_function(text)

        predictions.append(pred)

    acc = accuracy_score(test_labels, predictions)

    f1 = f1_score(test_labels, predictions, average="weighted")

    return acc, f1


# =============================
# 3 BERT 预测
# =============================

print("\nLoading BERT model...")

bert_model = load_model()
bert_model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def bert_predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = bert_model(**inputs)

        logits = outputs.logits

        pred = torch.argmax(logits, dim=1).item()

    return pred


# =============================
# 4 LoRA 预测
# =============================

print("\nLoading LoRA model...")

lora_model = load_lora_model()
lora_model.eval()


def lora_predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = lora_model(**inputs)

        logits = outputs.logits

        pred = torch.argmax(logits, dim=1).item()

    return pred


# =============================
# 5 Prompt 预测
# =============================

print("\nLoading Prompt LLM...")

prompt_llm = load_llm()


label_map = {
    "World": 0,
    "Sports": 1,
    "Business": 2,
    "Sci/Tech": 3
}


def prompt_predict(text):

    prompt = f"""
Classify the following news into one category:

World
Sports
Business
Sci/Tech

Text:
{text}

Return ONLY the category name.
Category:
"""

    result = prompt_llm(prompt, max_new_tokens=5)

    output = result[0]["generated_text"]

    for key in label_map:

        if key.lower() in output.lower():

            return label_map[key]

    return 0


# =============================
# 6 RAG 预测
# =============================

print("\nLoading RAG system...")

texts, labels = load_data()

index, embed_model = build_vector_index(texts)

rag_llm = load_llm()


def rag_predict(text):

    label, _ = rag_classify(
        text,
        texts,
        labels,
        index,
        embed_model,
        rag_llm
    )

    return int(label)


# =============================
# 7 运行评测
# =============================

results = {}

results["BERT"] = evaluate_model("BERT", bert_predict)

results["LoRA"] = evaluate_model("LoRA", lora_predict)

results["Prompt"] = evaluate_model("Prompt", prompt_predict)

results["RAG"] = evaluate_model("RAG", rag_predict)


# =============================
# 8 打印结果
# =============================

print("\n===== Final Results =====")

print(f"{'Model':<10} {'Accuracy':<10} {'F1':<10}")

print("-" * 30)

for model, (acc, f1) in results.items():

    print(f"{model:<10} {acc:.4f}     {f1:.4f}")

