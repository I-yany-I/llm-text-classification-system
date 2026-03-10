from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


def load_lora_model():

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, lora_config)

    print("LoRA model loaded!")

    return model


if __name__ == "__main__":

    model = load_lora_model()

    print(model)
