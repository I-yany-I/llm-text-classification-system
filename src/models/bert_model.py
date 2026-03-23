import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.paths import ensure_project_dirs, get_model_dir


class BertClassifier:

    def __init__(self, prefer_trained: bool = True):
        ensure_project_dirs()

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        model_path = get_model_dir("bert")

        # 自动判断是否有训练模型
        if prefer_trained and model_path.exists():

            print("Loading trained BERT model...")

            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path)
            )

        else:

            print("No trained model found. Loading base model...")

            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=4
            )

        self.model.eval()

    def predict(self, text):
        return self.predict_batch([text])[0]

    def predict_batch(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.argmax(outputs.logits, dim=1)

        return preds.tolist()

