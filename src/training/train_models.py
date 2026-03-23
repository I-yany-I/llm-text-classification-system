import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bert_model import BertClassifier
from src.models.lora_model import LoRAClassifier
from src.paths import ensure_project_dirs, get_checkpoint_dir, get_model_dir


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def load_tokenized_dataset():
    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset, tokenizer


def build_model(model_name: str):
    if model_name == "bert":
        return BertClassifier(prefer_trained=False).model
    if model_name == "lora":
        return LoRAClassifier(prefer_trained=False).model
    raise ValueError("model_name must be 'bert' or 'lora'")


def train_model(model_name: str, num_train_epochs: int = 1):
    ensure_project_dirs()
    print(f"\nTraining {model_name.upper()}...")

    dataset, tokenizer = load_tokenized_dataset()
    model = build_model(model_name)
    checkpoint_dir = get_checkpoint_dir(model_name)
    final_model_dir = get_model_dir(model_name)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print(f"Saved final {model_name.upper()} model to: {final_model_dir}")
    print(f"Saved checkpoints to: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train BERT and/or LoRA text classification models.")
    parser.add_argument("--model", choices=["bert", "lora", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    targets = ["bert", "lora"] if args.model == "all" else [args.model]
    for model_name in targets:
        train_model(model_name, num_train_epochs=args.epochs)


if __name__ == "__main__":
    main()
