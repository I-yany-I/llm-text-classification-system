import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictors import METHODS, get_model
from src.paths import PLOTS_DIR, PREDICTIONS_DIR, ensure_project_dirs


EVAL_SAMPLE_SIZE = 200


def evaluate_method(method: str, texts, labels):
    predictor = get_model(method)
    predictions = []

    for text in tqdm(texts, desc=f"Evaluating {method.upper()}"):
        predictions.append(predictor.predict(text))

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {
        "method": method,
        "accuracy": accuracy,
        "f1": f1,
        "predictions": predictions,
    }


def save_visualizations(results):
    methods = [item["method"].upper() for item in results]
    accuracies = [item["accuracy"] for item in results]
    f1_scores = [item["f1"] for item in results]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, accuracies, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.ylim(0, 1)
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_comparison.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(methods, f1_scores, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    plt.ylim(0, 1)
    plt.title("F1 Comparison")
    plt.ylabel("Weighted F1")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "f1_comparison.png")
    plt.close()


def save_metrics(results):
    summary = [
        {
            "method": item["method"],
            "accuracy": round(item["accuracy"], 4),
            "f1": round(item["f1"], 4),
        }
        for item in results
    ]

    with open(PREDICTIONS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    lines = ["method,accuracy,f1"]
    for item in summary:
        lines.append(f"{item['method']},{item['accuracy']},{item['f1']}")

    with open(PREDICTIONS_DIR / "evaluation_summary.csv", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    ensure_project_dirs()

    print("Loading AG News test split...")
    dataset = load_dataset("ag_news")
    test_texts = dataset["test"]["text"][:EVAL_SAMPLE_SIZE]
    test_labels = dataset["test"]["label"][:EVAL_SAMPLE_SIZE]

    results = []
    for method in METHODS:
        print(f"\nLoading {method.upper()}...")
        result = evaluate_method(method, test_texts, test_labels)
        results.append(result)
        print(
            f"{method.upper()} -> "
            f"Accuracy: {result['accuracy']:.4f}, "
            f"Weighted F1: {result['f1']:.4f}"
        )

    save_visualizations(results)
    save_metrics(results)

    print(f"\nSaved plots to: {PLOTS_DIR}")
    print(f"Saved metric summaries to: {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()
