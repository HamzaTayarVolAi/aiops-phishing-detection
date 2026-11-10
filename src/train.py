"""Train and compare two email classifiers on the same held-out data."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

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


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the combined Kaggle CSV and remove missing or ambiguous examples."""
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV not found at {path}. See README.md for the Kaggle download."
        )

    raw = pd.read_csv(path, dtype="string", keep_default_na=False)
    if not {TEXT, LABEL}.issubset(raw.columns):
        raise ValueError("Use the combined phishing_email.csv with text_combined and label columns.")

    emails = raw[[TEXT, LABEL]].copy()
    emails[TEXT] = emails[TEXT].str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    emails[LABEL] = emails[LABEL].str.strip()
    present = emails.loc[emails[TEXT].ne("") & emails[LABEL].ne("")]
    if not present[LABEL].isin(["0", "1"]).all():
        raise ValueError("Expected label 0 for legitimate email or 1 for spam/phishing.")

    # Exclude all copies when the same normalized text has conflicting labels.
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
            "Removed for conflicting labels": int(conflicting.sum()),
            "Removed as duplicate text": len(unambiguous) - len(cleaned),
            "Rows after cleaning": len(cleaned),
        }
    )
    return cleaned, audit


def split_dataset(
    emails: pd.DataFrame,
) -> tuple[tuple[pd.Series, pd.Series, pd.Series, pd.Series], pd.DataFrame]:
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


def compare_models(
    emails: pd.DataFrame, *, baseline_only: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the baseline and optionally the second classifier on the same split."""
    split, split_counts = split_dataset(emails)
    text_train, text_test, label_train, label_test = split

    classifiers: dict[str, MultinomialNB | LogisticRegression] = {
        "Naive Bayes (baseline)": MultinomialNB(),
    }
    if not baseline_only:
        classifiers["Logistic regression"] = LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=SEED
        )
    rows = []
    for name, classifier in classifiers.items():
        model = make_pipeline(TfidfVectorizer(min_df=2, max_features=30_000), classifier)
        # The pipeline learns vocabulary and TF-IDF weights from training data only.
        model.fit(text_train, label_train)
        predicted = model.predict(text_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            label_test, predicted, average="binary", pos_label=1, zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(label_test, predicted, labels=[0, 1]).ravel()
        rows.append(
            {
                "model": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            }
        )

    return split_counts, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to phishing_email.csv.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--split-only", action="store_true", help="Show the data cleaning and train/test split counts."
    )
    mode.add_argument(
        "--baseline-only", action="store_true", help="Train and evaluate the Naive Bayes baseline."
    )
    args = parser.parse_args()
    metrics_path = BASELINE_METRICS_PATH if args.baseline_only else METRICS_PATH
    if args.data.resolve() == metrics_path.resolve():
        parser.error(f"Choose a data file other than {metrics_path.relative_to(ROOT)}.")

    metrics = None
    try:
        emails, audit = load_dataset(args.data)
        if args.split_only:
            _, split_counts = split_dataset(emails)
        else:
            split_counts, metrics = compare_models(emails, baseline_only=args.baseline_only)
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
    if metrics is not None:
        if args.baseline_only:
            print("\nBaseline on the test split")
        else:
            print("\nModel comparison on the test split")
        print(metrics.round(4).to_string(index=False))
        print(f"\nMetrics saved to {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
