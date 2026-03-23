from src.training.train_models import main, train_model

__all__ = ["main", "train_model"]


def train(model_type="bert"):
    train_model(model_type)


if __name__ == "__main__":
    main()
