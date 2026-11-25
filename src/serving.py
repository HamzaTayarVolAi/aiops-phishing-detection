"""Load a model saved by train.py and classify single emails at once."""

import json
import time
from pathlib import Path

import joblib

from train import MODELS, MODELS_DIR, CascadeClassifier, normalize_text

LABEL_NAMES = {0: "legitimate", 1: "spam or phishing"}


def load_model(key: str):
    """Return (display name, model) for a model saved in models/<key>/."""
    if key not in MODELS:
        raise ValueError(f"Unknown model '{key}'. Choose one of: {', '.join(MODELS)}")
    folder = MODELS_DIR / key
    if not (folder / "model.json").is_file():
        raise FileNotFoundError(
            f"No saved model in {folder}. Train it first: python src/train.py --models {key}"
        )
    name = json.loads((folder / "model.json").read_text("utf-8"))["name"]

    if key in ("nb", "lr"):
        return name, joblib.load(folder / "model.joblib")
    if key == "cnn":
        from deep_models import TextCNNClassifier

        return name, TextCNNClassifier.load(folder)
    if key == "bert":
        from deep_models import DistilBertClassifier

        return name, DistilBertClassifier.load(folder)
    if key == "cascade":
        from deep_models import DistilBertClassifier

        fast = joblib.load(folder / "fast_model.joblib")
        slow = DistilBertClassifier.load(folder / "distilbert")
        return name, CascadeClassifier(fast, slow, verbose=False)
    raise ValueError(f"Loading '{key}' is not supported.")


def model_device(model) -> str:
    """Where the model runs: "cuda" for the NVIDIA GPU, otherwise "cpu"."""
    neural = getattr(model, "slow_model", model)  # the cascade's neural part
    device = getattr(neural, "device", None)
    return device.type if device is not None else "cpu"


def spam_probability(model, text: str) -> float:
    """Probability that one already normalized email is spam or phishing."""
    probabilities = model.predict_proba([text])
    # scikit-learn returns one column per class; our neural models return one value per email
    return float(probabilities[0][1] if getattr(probabilities, "ndim", 1) == 2 else probabilities[0])


def classify(model, raw_text: str) -> dict:
    """Clean one email like the training data, classify it and time the prediction."""
    text = normalize_text(raw_text)
    if not text:
        raise ValueError("The email text is empty.")
    started = time.perf_counter()
    probability = spam_probability(model, text)
    milliseconds = (time.perf_counter() - started) * 1000
    label = int(probability >= 0.5)
    return {
        "label": label,
        "label_name": LABEL_NAMES[label],
        "spam_probability": round(probability, 4),
        "milliseconds": round(milliseconds, 2),
    }


if __name__ == "__main__":
    # Quick check without the API: python src/serving.py lr "verify your account now"
    import sys

    if len(sys.argv) < 3:
        sys.exit('Usage: python src/serving.py <nb|lr|cnn|bert|cascade> "email text"')
    try:
        started = time.perf_counter()
        model_name, loaded = load_model(sys.argv[1])
        print(
            f"Loaded {model_name} in {time.perf_counter() - started:.1f} s "
            f"on {model_device(loaded)}"
        )
        classify(loaded, "warm-up email")  # the first prediction on a GPU is slower
        print(classify(loaded, " ".join(sys.argv[2:])))
    except (OSError, ValueError) as error:
        sys.exit(f"Error: {error}")