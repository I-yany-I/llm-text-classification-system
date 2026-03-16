from transformers import AutoTokenizer
from datasets import load_dataset


def load_tokenizer():
    """
    加载 BERT tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer


def tokenize_dataset():
    """
    对数据集进行tokenize
    """

    dataset = load_dataset("ag_news")
    tokenizer = load_tokenizer()

    def tokenize(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)

    print("Tokenization finished!")

    print("\nExample tokenized sample:")
    print(tokenized_dataset["train"][0])

    return tokenized_dataset


if __name__ == "__main__":
    tokenize_dataset()
