from src.models.bert_model import BertClassifier
from src.models.lora_model import LoRAClassifier
from src.paths import get_model_dir
from src.training.train_models import train_model


def load_or_train_bert():

    model_path = get_model_dir("bert")

    if not model_path.exists():

        print("BERT model not found. Training...")

        train_model("bert")

    else:
        print("Loading existing BERT model...")

    return BertClassifier()


def load_or_train_lora():

    model_path = get_model_dir("lora")

    if not model_path.exists():

        print("LoRA model not found. Training...")

        train_model("lora")

    else:
        print("Loading existing LoRA model...")

    return LoRAClassifier()
