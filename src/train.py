from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer
from model import load_model
from lora_model import load_lora_model


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

    return tokenized_dataset


def train():

    dataset = prepare_dataset()

# --------------------------------------------------------
    # 1 Fine-tuning(Baseline BERT)
    # model = load_model()

# -------------------------------------------------------------
    # 2 loRA Fine-tuning
    model = load_lora_model()

# --------------------------------------------------------------

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    trainer.train()


if __name__ == "__main__":
    train()
