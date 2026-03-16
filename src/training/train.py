from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer
from lora_model import load_lora_model
from sklearn.metrics import f1_score, accuracy_score

def prepare_dataset():
    dataset = load_dataset("ag_news")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)
    # 设置 PyTorch tensor 格式
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return tokenized_dataset

# --------------------------------------
# 计算 F1 和 Accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# --------------------------------------
def train():
    dataset = prepare_dataset()

    # 1 Fine-tuning(Baseline BERT)
    # model = load_model()

    # 2 loRA Fine-tuning
    model = load_lora_model()

    training_args = TrainingArguments(
        output_dir="../results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        save_strategy="epoch",        # 每轮保存一次模型
        logging_steps=50,             # 日志间隔
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    # 最终评估
    results = trainer.evaluate()
    print("Evaluation Results:", results)

if __name__ == "__main__":
    train()
