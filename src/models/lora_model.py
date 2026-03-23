import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from src.paths import ensure_project_dirs, get_model_dir


class LoRAClassifier:

    def __init__(self, prefer_trained: bool = True):
        ensure_project_dirs()

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        base_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=4
        )

        model_path = get_model_dir("lora")

        if prefer_trained and model_path.exists():

            print("Loading trained LoRA model...")

            self.model = PeftModel.from_pretrained(
                base_model,
                str(model_path)
            )

        else:

            print("No trained LoRA found. Initializing new LoRA adapter...")

            config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["query", "value"],
            )
            self.model = get_peft_model(base_model, config)

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

        pred = torch.argmax(outputs.logits, dim=1)

        return pred.tolist()
