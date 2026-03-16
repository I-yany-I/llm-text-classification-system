from datasets import load_dataset


def load_ag_news():
    """
    加载 AG News 数据集
    """
    dataset = load_dataset("ag_news")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    print("\nExample:")
    print(train_dataset[0])

    return train_dataset, test_dataset


if __name__ == "__main__":
    load_ag_news()
