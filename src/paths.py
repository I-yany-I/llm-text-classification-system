from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"


def get_model_dir(model_type: str) -> Path:
    return MODELS_DIR / model_type


def get_checkpoint_dir(model_type: str) -> Path:
    return CHECKPOINTS_DIR / model_type


def ensure_project_dirs() -> None:
    for path in (
        ARTIFACTS_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR,
        PLOTS_DIR,
        PREDICTIONS_DIR,
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        VECTOR_STORE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
