import argparse
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

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


def train_naive_bayes(text_train, label_train):
    """Fit the TF-IDF and multinomial Naive Bayes pipeline on training data."""
    model = make_pipeline(
        TfidfVectorizer(min_df=TFIDF_MIN_DF, max_features=TFIDF_MAX_FEATURES),
        MultinomialNB(),
    )
    return model.fit(text_train, label_train)


def train_logistic_regression(text_train, label_train):
    """Fit the TF-IDF and logistic regression pipeline on training data."""
    model = make_pipeline(
        TfidfVectorizer(min_df=TFIDF_MIN_DF, max_features=TFIDF_MAX_FEATURES),
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=SEED),
    )
    return model.fit(text_train, label_train)


def log_mlflow_run(
    name, model, scores, matrix, *, data_path: Path, dataset_sha256: str
):
    """Save one model's parameters, test results, and fitted pipeline in MLflow."""
    tfidf = model[0]
    classifier = model[-1]

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "dataset": data_path.name,
                "dataset_sha256": dataset_sha256,
                "split": "stratified",
                "test_size": TEST_SIZE,
                "seed": SEED,
                "tfidf.min_df": tfidf.min_df,
                "tfidf.max_features": tfidf.max_features,
                "classifier": type(classifier).__name__,
                **{
                    f"classifier.{key}": value
                    for key, value in classifier.get_params(deep=False).items()
                },
            }
        )
        mlflow.log_metrics(scores)
        mlflow.log_dict(
            {"labels": [0, 1], "actual_by_predicted": matrix.tolist()},
            "confusion_matrix.json",
        )
        mlflow.sklearn.log_model(model, name="pipeline")


def compare_models(
    emails: pd.DataFrame, *, data_path: Path, baseline_only: bool = False
):
    """Evaluate the selected pipelines on one shared test split."""
    split, split_counts = split_dataset(emails)
    text_train, text_test, label_train, label_test = split

    trainers = {"Naive Bayes (baseline)": train_naive_bayes}
    if not baseline_only:
        trainers["Logistic regression"] = train_logistic_regression

    with data_path.open("rb") as source:
        dataset_sha256 = file_digest(source, "sha256").hexdigest()

    rows = []
    for name, train_model in trainers.items():
        model = train_model(text_train, label_train)
        predicted = model.predict(text_test)

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
        }
        log_mlflow_run(
            name, model, scores, matrix, data_path=data_path, dataset_sha256=dataset_sha256
        )
        rows.append({"model": name, **scores})

    return split_counts, pd.DataFrame(rows)


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
    args = parser.parse_args()

    metrics_path = BASELINE_METRICS_PATH if args.baseline_only else METRICS_PATH
    if args.data.resolve() == metrics_path.resolve():
        parser.error(f"Choose a data file other than {metrics_path.relative_to(ROOT)}.")

    try:
        emails, audit = load_dataset(args.data)

        if args.split_only:
            _, split_counts = split_dataset(emails)
        else:
            split_counts, metrics = compare_models(
                emails, data_path=args.data, baseline_only=args.baseline_only
            )

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
