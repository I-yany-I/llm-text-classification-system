from transformers import AutoModelForSequenceClassification


def load_model():
    """
    加载 BERT 文本分类模型
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4  # AG News 有4个类别
    )

    print("Model loaded successfully!")

    return model


if __name__ == "__main__":
    model = load_model()

    print(model)
