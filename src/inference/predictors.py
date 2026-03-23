from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.models.bert_model import BertClassifier
from src.models.lora_model import LoRAClassifier
from src.prompt.prompt_classifier import classify_news as prompt_classify
from src.prompt.prompt_classifier import load_llm as load_prompt_llm
from src.rag.rag_pipeline import build_vector_index, load_data, load_llm as load_rag_llm, rag_classify


LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

METHODS = ["bert", "lora", "prompt", "rag"]


@dataclass
class PredictionResult:
    method: str
    label_id: int
    label_name: str
    raw_output: str
    examples: List[str]


def label_name_from_id(label_id: int) -> str:
    return LABEL_MAP.get(label_id, "Unknown")


def parse_batch_texts(raw_text: str) -> List[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


class BertPredictor:
    def __init__(self):
        self.model = BertClassifier()

    def predict(self, text: str) -> int:
        return self.model.predict(text)

    def predict_batch(self, texts: List[str]) -> List[int]:
        return self.model.predict_batch(texts)

    def predict_with_details(self, text: str) -> PredictionResult:
        label_id = self.predict(text)
        return PredictionResult("bert", label_id, label_name_from_id(label_id), str(label_id), [])


class LoRAPredictor:
    def __init__(self):
        self.model = LoRAClassifier()

    def predict(self, text: str) -> int:
        return self.model.predict(text)

    def predict_batch(self, texts: List[str]) -> List[int]:
        return self.model.predict_batch(texts)

    def predict_with_details(self, text: str) -> PredictionResult:
        label_id = self.predict(text)
        return PredictionResult("lora", label_id, label_name_from_id(label_id), str(label_id), [])


class PromptPredictor:
    def __init__(self):
        self.llm = load_prompt_llm()

    def predict(self, text: str) -> int:
        return prompt_classify(text, self.llm)

    def predict_batch(self, texts: List[str]) -> List[int]:
        return [self.predict(text) for text in texts]

    def predict_with_details(self, text: str) -> PredictionResult:
        label_id = self.predict(text)
        return PredictionResult("prompt", label_id, label_name_from_id(label_id), str(label_id), [])


class RAGPredictor:
    def __init__(self):
        texts, labels = load_data()
        self.texts = texts
        self.labels = labels
        self.index, self.embed_model = build_vector_index(texts)
        self.llm = load_rag_llm()

    def predict_with_details(self, text: str) -> PredictionResult:
        label, examples = rag_classify(
            text,
            self.texts,
            self.labels,
            self.index,
            self.embed_model,
            self.llm,
        )
        try:
            label_id = int(label)
        except ValueError:
            label_id = -1

        return PredictionResult("rag", label_id, label_name_from_id(label_id), str(label), examples)

    def predict(self, text: str) -> int:
        return self.predict_with_details(text).label_id

    def predict_batch(self, texts: List[str]) -> List[int]:
        return [self.predict(text) for text in texts]


_MODEL_CACHE: Dict[str, object] = {}


def get_model(method: str):
    normalized_method = method.lower()
    if normalized_method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")

    if normalized_method not in _MODEL_CACHE:
        factories = {
            "bert": BertPredictor,
            "lora": LoRAPredictor,
            "prompt": PromptPredictor,
            "rag": RAGPredictor,
        }
        _MODEL_CACHE[normalized_method] = factories[normalized_method]()

    return _MODEL_CACHE[normalized_method]
