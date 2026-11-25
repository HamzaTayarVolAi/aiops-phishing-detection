import argparse
import time
from hashlib import file_digest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
from urllib.request import url2pathname

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline

from deep_models import DistilBertClassifier, TextCNNClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "phishing_email.csv"
METRICS_PATH = ROOT / "results" / "metrics.csv"
BASELINE_METRICS_PATH = ROOT / "results" / "baseline_metrics.csv"

TEXT = "text_combined"
LABEL = "label"
TEST_SIZE = 0.20
SEED = 42
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 30_000
EXPERIMENT_NAME = "Spam and Phishing Email Detection"


def load_dataset(path: Path):
    """Load the CSV and remove missing, duplicate, or ambiguous examples."""
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV not found at {path}. See README.md for the Kaggle download."
        )

    raw = pd.read_csv(path, dtype="string", keep_default_na=False)
    if not {TEXT, LABEL}.issubset(raw.columns):
        raise ValueError(
            "Use the combined phishing_email.csv with text_combined and label columns."
        )

    emails = raw[[TEXT, LABEL]].copy()
    emails[TEXT] = (
        emails[TEXT].str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    )
    emails[LABEL] = emails[LABEL].str.strip()

    present = emails.loc[emails[TEXT].ne("") & emails[LABEL].ne("")]
    if not present[LABEL].isin(["0", "1"]).all():
        raise ValueError("Expected label 0 for legitimate email or 1 for spam/phishing.")

    # Exclude all copies of a text when its labels conflict.
    conflicting = present.groupby(TEXT, sort=False)[LABEL].transform("nunique").gt(1)
    unambiguous = present.loc[~conflicting]
    cleaned = unambiguous.drop_duplicates(subset=TEXT).copy()
    cleaned[LABEL] = cleaned[LABEL].astype(int)

    if cleaned[LABEL].nunique() != 2:
        raise ValueError("The cleaned dataset must contain both email classes.")

    audit = pd.Series(
        {
            "Rows read": len(raw),
            "Removed for missing text or label": len(raw) - len(present),
            "Removed for conflicting labels": conflicting.sum(),
            "Removed as duplicate text": len(unambiguous) - len(cleaned),
            "Rows after cleaning": len(cleaned),
        }
    )
    return cleaned, audit


def split_dataset(emails: pd.DataFrame):
    """Create a reproducible split and count each class in both parts."""
    text_train, text_test, label_train, label_test = train_test_split(
        emails[TEXT],
        emails[LABEL],
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=emails[LABEL],
    )

    if label_train.nunique() != 2 or label_test.nunique() != 2:
        raise ValueError("The split needs examples of both classes in training and test data.")

    split_counts = pd.DataFrame(
        {
            "Cleaned": emails[LABEL].value_counts(),
            "Train": label_train.value_counts(),
            "Test": label_test.value_counts(),
        },
        index=[0, 1],
    )
    split_counts.index = pd.Index(["0 (legitimate)", "1 (spam/phishing)"], name="Class")

    return (text_train, text_test, label_train, label_test), split_counts


def train_naive_bayes(text_train, label_train, **_):
    """Fit the TF-IDF and multinomial Naive Bayes pipeline on training data."""
    model = make_pipeline(
        TfidfVectorizer(min_df=TFIDF_MIN_DF, max_features=TFIDF_MAX_FEATURES),
        MultinomialNB(),
    )
    return model.fit(text_train, label_train)


def train_logistic_regression(text_train, label_train, **_):
    """Fit the TF-IDF and logistic regression pipeline on training data."""
    model = make_pipeline(
        TfidfVectorizer(min_df=TFIDF_MIN_DF, max_features=TFIDF_MAX_FEATURES),
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=SEED),
    )
    return model.fit(text_train, label_train)


def train_cnn(text_train, label_train, **_):
    """Fit a word tokenizer and a text CNN; validation comes from the training data."""
    return TextCNNClassifier(seed=SEED).fit(text_train, label_train)


def train_distilbert(text_train, label_train, *, bert_train_limit=None, bert_max_tokens=256, **_):
    """Fine-tune DistilBERT; validation comes from the training data."""
    return DistilBertClassifier(
        seed=SEED, train_limit=bert_train_limit, max_tokens=bert_max_tokens
    ).fit(text_train, label_train)


# Command-line key -> (name in results and MLflow, training function, extra packages)
MODELS = {
    "nb": ("Naive Bayes (baseline)", train_naive_bayes, []),
    "lr": ("Logistic regression", train_logistic_regression, []),
    "cnn": ("CNN (word tokenizer)", train_cnn, ["torch"]),
    "bert": ("DistilBERT (fine-tuned)", train_distilbert, ["torch", "transformers"]),
}
MODEL_ORDER = [name for name, _, _ in MODELS.values()]


def check_packages(model_keys):
    """Fail before training starts when a selected model lacks its packages."""
    missing = sorted(
        {package for key in model_keys for package in MODELS[key][2] if find_spec(package) is None}
    )
    if missing:
        raise ValueError(
            f"Missing package(s): {', '.join(missing)}. "
            "Run: python -m pip install -r requirements.txt"
        )


def check_mlflow_artifacts():
    """Fail before training when MLflow would save artifacts to a folder this PC cannot use.

    An mlflow.db created on another computer keeps that computer's artifact folder.
    """
    location = mlflow.set_experiment(EXPERIMENT_NAME).artifact_location
    parsed = urlparse(location)
    if parsed.scheme == "file":
        folder = Path(url2pathname(parsed.path))
    elif len(parsed.scheme) <= 1:  # plain path, including Windows drive letters
        folder = Path(location)
    else:
        return  # remote storage, such as s3://
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(
            f"MLflow saves artifacts of '{EXPERIMENT_NAME}' to {folder}, which this PC "
            f"cannot use ({error.strerror}). The tracking database was probably created on "
            "another computer. Use a new database file, for example: "
            'set MLFLOW_TRACKING_URI=sqlite:///mlflow_local.db'
        ) from error


def model_params(model) -> dict:
    """Describe a fitted model for MLflow."""
    if isinstance(model, Pipeline):
        tfidf = model[0]
        classifier = model[-1]
        return {
            "features": "TF-IDF",
            "tfidf.min_df": tfidf.min_df,
            "tfidf.max_features": tfidf.max_features,
            "classifier": type(classifier).__name__,
            **{
                f"classifier.{key}": value
                for key, value in classifier.get_params(deep=False).items()
            },
        }
    return model.params()


def log_mlflow_run(name, model, scores, matrix, *, data_path: Path, dataset_sha256: str):
    """Save one model's parameters, test results, and fitted model in MLflow."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "dataset": data_path.name,
                "dataset_sha256": dataset_sha256,
                "split": "stratified",
                "test_size": TEST_SIZE,
                "seed": SEED,
                **model_params(model),
            }
        )
        mlflow.log_metrics(scores)
        if hasattr(model, "extra_metrics"):
            mlflow.log_metrics(model.extra_metrics())
        mlflow.log_dict(
            {"labels": [0, 1], "actual_by_predicted": matrix.tolist()},
            "confusion_matrix.json",
        )
        if isinstance(model, Pipeline):
            mlflow.sklearn.log_model(model, name="pipeline")
        else:
            model.log_to_mlflow(mlflow)


def compare_models(
    emails: pd.DataFrame, *, data_path: Path, model_keys, bert_train_limit=None, bert_max_tokens=256
):
    """Evaluate the selected models on one shared test split."""
    check_packages(model_keys)
    check_mlflow_artifacts()
    split, split_counts = split_dataset(emails)
    text_train, text_test, label_train, label_test = split

    with data_path.open("rb") as source:
        dataset_sha256 = file_digest(source, "sha256").hexdigest()

    rows = []
    for key in model_keys:
        name, train_model, _ = MODELS[key]
        print(f"Training {name} ...", flush=True)

        started = time.perf_counter()
        model = train_model(
            text_train,
            label_train,
            bert_train_limit=bert_train_limit,
            bert_max_tokens=bert_max_tokens,
        )
        train_seconds = time.perf_counter() - started

        started = time.perf_counter()
        predicted = model.predict(text_test)
        predict_seconds = time.perf_counter() - started

        precision, recall, f1, _ = precision_recall_fscore_support(
            label_test, predicted, average="binary", pos_label=1, zero_division=0
        )
        matrix = confusion_matrix(label_test, predicted, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()

        scores = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "train_seconds": round(train_seconds, 1),
            "predict_seconds": round(predict_seconds, 1),
        }
        log_mlflow_run(
            name, model, scores, matrix, data_path=data_path, dataset_sha256=dataset_sha256
        )
        rows.append({"model": name, **scores})
        print(f"Done: recall {recall:.4f}, F1 {f1:.4f} ({train_seconds / 60:.1f} min)\n")

    return split_counts, pd.DataFrame(rows)


def merge_metrics(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Replace the rows of the models just trained and keep the other current models' rows.

    Rows of models that are no longer part of the project are dropped.
    """
    if not path.is_file():
        return new
    previous = pd.read_csv(path)
    kept = previous.loc[
        ~previous["model"].isin(new["model"]) & previous["model"].isin(MODEL_ORDER)
    ]
    columns = list(dict.fromkeys([*new.columns, *previous.columns]))
    merged = pd.concat([kept, new], ignore_index=True)[columns]
    rank = {name: index for index, name in enumerate(MODEL_ORDER)}
    return merged.sort_values(
        "model", key=lambda names: names.map(rank).fillna(len(rank)), kind="stable"
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=DATA_PATH, help="Path to phishing_email.csv."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--split-only", action="store_true", help="Show the data cleaning and split counts."
    )
    mode.add_argument(
        "--baseline-only", action="store_true", help="Train and evaluate only Naive Bayes."
    )
    mode.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        help="Models to train (default: all). Other rows in results/metrics.csv are kept.",
    )
    parser.add_argument(
        "--bert-train-limit",
        type=int,
        metavar="N",
        help="Fine-tune DistilBERT on a stratified sample of N training emails (for a CPU).",
    )
    parser.add_argument(
        "--bert-max-tokens",
        type=int,
        default=256,
        metavar="N",
        help="Tokens per email DistilBERT reads (default 256; 128 trains about twice as fast).",
    )
    args = parser.parse_args()
    if not 16 <= args.bert_max_tokens <= 512:
        parser.error("--bert-max-tokens must be between 16 and 512.")
    if args.bert_train_limit is not None and args.bert_train_limit < 100:
        parser.error("--bert-train-limit must be at least 100.")

    if args.baseline_only:
        model_keys = ["nb"]
    else:
        model_keys = list(dict.fromkeys(args.models or MODELS))

    metrics_path = BASELINE_METRICS_PATH if args.baseline_only else METRICS_PATH
    if args.data.resolve() == metrics_path.resolve():
        parser.error(f"Choose a data file other than {metrics_path.relative_to(ROOT)}.")

    try:
        emails, audit = load_dataset(args.data)

        if args.split_only:
            _, split_counts = split_dataset(emails)
        else:
            split_counts, metrics = compare_models(
                emails,
                data_path=args.data,
                model_keys=model_keys,
                bert_train_limit=args.bert_train_limit,
                bert_max_tokens=args.bert_max_tokens,
            )
            if not args.baseline_only:
                metrics = merge_metrics(metrics, metrics_path)

            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=metrics_path.parent) as temp_dir:
                temp_metrics = Path(temp_dir) / metrics_path.name
                metrics.to_csv(temp_metrics, index=False, float_format="%.6f")
                temp_metrics.replace(metrics_path)

    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")

    print("Dataset cleaning")
    print(audit.to_string())
    print(f"\nStratified 80/20 split (seed {SEED})")
    print(split_counts.to_string())

    if not args.split_only:
        title = "Baseline" if args.baseline_only else "Model comparison"
        print(f"\n{title} on the test split")
        print(metrics.round(4).to_string(index=False))
        print(f"\nMetrics saved to {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()