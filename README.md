# Spam and Phishing Email Detection

<<<<<<< HEAD
This AIOps project reads email text and predicts one of two classes. Label `0` means legitimate email. Label `1` means spam or phishing. Employees and IT teams can use the result to decide which messages need a closer check.

## Dataset

We use `phishing_email.csv` from the *Phishing Email Dataset* on Kaggle, published by Naser Abdullah Alam with Amith Khandakar. The file has 82,486 rows. `text_combined` contains the email text, and `label` contains the class. [Dataset DOI](https://doi.org/10.34740/kaggle/ds/5074342) · [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/)
=======
This AIOps project reads email text and predicts one of two classes. Label 0 means legitimate email. Label 1 means spam or phishing. Employees and IT teams can use the result to choose which messages need a closer check.

## Business problem

| Question                       | Answer                                                      |
| ------------------------------ | ----------------------------------------------------------- |
| Who uses the result?           | The IT security team, which reviews suspicious emails       |
| What decision does it support? | Deliver an email normally, or flag it for review            |
| Prediction target              | Label 0 (legitimate) or 1 (spam or phishing) for each email |
| Operational setting            | On demand: one prediction for each incoming email           |
| Main metric                    | Recall for label 1, with precision as a guardrail           |

The two error types have different costs:

- A **false negative** is a spam or phishing email marked as legitimate. It reaches the inbox without a check. One click on a phishing link can lead to stolen passwords or malware.
- A **false positive** is a legitimate email marked as suspicious. It costs review time and delays the email, but nothing is lost.

A missed phishing email usually costs more than an extra review. For this reason we focus on **recall**. Precision must stay high so that the review team is not flooded with legitimate emails.

## Dataset

We use `phishing_email.csv` from the [Phishing Email Dataset on Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset), published by Naser Abdullah Alam with Amith Khandakar. The file has 82,486 rows. `text_combined` contains the email text, and `label` contains the correct class. Dataset DOI · CC BY-SA 4.0 license
>>>>>>> a6a29023fba16536b99de7152966b631d987e1bb

| Label | Meaning          | Rows in the original CSV | Suggested action |
| ----- | ---------------- | ------------------------ | ---------------- |
| 0     | Legitimate email | 39,595                   | Deliver normally |
| 1     | Spam or phishing | 42,891                   | Flag for review  |

The labels are already in the dataset. Spam and phishing use the same label. Both classes are of similar size, so the data is nearly balanced.

The data card asks users to cite Al-Subaiey et al. (2024), _Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection_. Paper

### Data cleaning

`src/train.py` cleans the data before it splits it:

1. Replace repeated spaces and line breaks with one space, and convert the text to lower case.
2. Remove rows with empty text or an empty label.
3. If the same text appears with label 0 and label 1, remove all copies of that text. We cannot know which label is correct.
4. Keep only one copy of each repeated text, so the same email cannot appear in both the training set and the test set.

In our reference run, 82,077 emails remained after cleaning, so 409 rows were removed. The script prints how many rows each step removed.

### Split logic

- **80% training, 20% test**, stratified by label. Both sets keep the same share of each class.
- **Seed 42** makes the split repeatable with the same input data.
- We use a random split because the dataset has no timestamps. We cannot test on newer emails than those used for training.
- TF-IDF sits inside the pipeline. It learns its vocabulary and word weights from the training emails only, so no test information leaks into training.

| Class             | Cleaned           | Train             | Test       |
| ----------------- | ----------------- | ----------------- | ---------- |
| 0 (legitimate)    | see script output | see script output | 7,836      |
| 1 (spam/phishing) | see script output | see script output | 8,580      |
| **Total**         | **82,077**        | **65,661**        | **16,416** |

## Setup and training

Install Python 3.12. Download `phishing_email.csv` and put it in `data/phishing_email.csv`. If the download is a ZIP, extract the CSV first. [Kaggle download](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)

Open PowerShell in the main project folder. These commands create a Python environment, install the required packages, and run training.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/train.py
```

If the environment is already active and the packages are installed, run `python src/train.py`.

<<<<<<< HEAD
## What `train.py` does

1. Read the CSV. Normalize whitespace and letter case. Remove rows with missing text or labels, repeated texts, and texts with conflicting labels.
2. Split the cleaned emails into 80% training and 20% testing. Stratification keeps the class proportions similar in both sets. Seed `42` makes the split repeatable with the same input data.
3. Train both models on the same training emails. Each model learns TF-IDF and the classifier from the training set.
=======
The script has three modes:

| Command                               | What it does                                             | Output                         |
| ------------------------------------- | -------------------------------------------------------- | ------------------------------ |
| `python src/train.py --split-only`    | Cleans the data and shows the class counts in each split | Console only                   |
| `python src/train.py --baseline-only` | Trains and evaluates only the Naive Bayes baseline       | `results/baseline_metrics.csv` |
| `python src/train.py`                 | Trains and compares both models                          | `results/metrics.csv`          |

Use `--data <path>` to read the CSV from another location.

## What train.py does

1. Read the original CSV. Clean spaces and letter case. Remove empty texts, repeated texts, and texts with different labels for the same message.
2. Put 80% of the emails in the training set and 20% in the test set. Both sets contain legitimate and spam or phishing emails. Seed 42 makes the split repeatable with the same input data.
3. Train two models on the same training emails. Each model learns from the email text and its correct label.
>>>>>>> a6a29023fba16536b99de7152966b631d987e1bb
4. Ask both models to predict the classes of the test emails using only their text. Compare the predictions with the known labels.
5. Print the results and save the scores in `results/metrics.csv`.

Both models use the same test set, so we can compare their results directly.

## The two models

<<<<<<< HEAD
| Model | Purpose |
| --- | --- |
| TF-IDF and multinomial Naive Bayes | Our baseline for the comparison |
| TF-IDF and logistic regression | The second model we compare with the baseline |
=======
| Model                              | Purpose                                                     |
| ---------------------------------- | ----------------------------------------------------------- |
| TF-IDF and multinomial Naive Bayes | Our baseline, which gives us a first result to compare with |
| TF-IDF and logistic regression     | The improved model that we compare with the baseline        |
>>>>>>> a6a29023fba16536b99de7152966b631d987e1bb

TF-IDF stands for Term Frequency and Inverse Document Frequency. It turns email text into numbers based on how often words appear. Words that appear in many emails usually get less weight. The classifier uses these numbers to find patterns.

<<<<<<< HEAD
A pipeline joins TF-IDF and a classifier into one object. The training script fits both pipelines and saves their evaluation scores in `results/metrics.csv`.
=======
A pipeline joins TF-IDF and a classifier into one object. During training, each pipeline learns its word rules from the training emails.

Both models use the same text features, so the comparison only changes the classifier:

| Setting               | Value                                                    |
| --------------------- | -------------------------------------------------------- |
| TF-IDF `min_df`       | 2 (a word must appear in at least 2 training emails)     |
| TF-IDF `max_features` | 30,000                                                   |
| Naive Bayes           | scikit-learn defaults (`alpha=1.0`)                      |
| Logistic regression   | `solver="liblinear"`, `max_iter=1000`, `random_state=42` |

**Why Naive Bayes is the baseline:** It is simple, fast and a common first model for text classification. It needs no tuning, so it gives a fair starting point.

**Why logistic regression is the improved model:** It is still simple and easy to explain, because each word gets a weight that we can inspect. Unlike Naive Bayes, it does not assume that words are independent of each other.

The script creates and trains both pipelines in the computer's memory while it runs. The scores in `results/metrics.csv` show how they performed. A saved pipeline file in `models/` holds what the chosen model learned, so the API can use it after the training script ends.
>>>>>>> a6a29023fba16536b99de7152966b631d987e1bb

## Results

The CSV contained 82,486 rows. Cleaning removed one row with missing text or a missing label and 408 duplicate texts, leaving 82,077 emails.

<<<<<<< HEAD
| Class | After cleaning | Training | Test |
| --- | ---: | ---: | ---: |
| Legitimate | 39,233 | 31,386 | 7,847 |
| Spam or phishing | 42,844 | 34,275 | 8,569 |
| **Total** | **82,077** | **65,661** | **16,416** |

Precision tells us what share of the emails marked as suspicious really were spam or phishing. Recall tells us what share of all spam or phishing emails the model found. F1 combines precision and recall into one score.

A false positive is a legitimate email marked as suspicious. A false negative is a spam or phishing email marked as legitimate.

| Model with TF-IDF | Precision | Recall | F1 | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes (baseline) | 98.41% | 96.81% | 97.61% | 134 | 273 |
| Logistic regression | 98.29% | 99.02% | 98.65% | 148 | 84 |

On the same test split, logistic regression missed 189 fewer spam or phishing emails and flagged 14 more legitimate emails for review.

These scores describe this public dataset. Similar email templates may appear in both sets, and emails in a real company may differ from those in the dataset.
=======
Precision tells us how many emails marked as suspicious really belong to label 1. Recall tells us how many of the suspicious emails the model found. F1 combines precision and recall into one score.

A false positive is a legitimate email marked as suspicious. A false negative is a spam or phishing email marked as legitimate.

The following reference run used 65,661 training emails and 16,416 test emails after cleaning. The test set contains 8,580 spam or phishing emails and 7,836 legitimate emails.

| Model with TF-IDF                  | Precision | Recall     | F1         | False positives | False negatives | Total errors |
| ---------------------------------- | --------- | ---------- | ---------- | --------------- | --------------- | ------------ |
| Multinomial Naive Bayes (baseline) | 98.41%    | 96.83%     | 97.61%     | 134             | 272             | 406          |
| Logistic regression                | 98.29%    | **99.02%** | **98.65%** | 148             | **84**          | **232**      |

### Comparison with the baseline

- **Logistic regression is better on our main metric.** Recall rises from 96.83% to 99.02%. The model missed 84 spam or phishing emails instead of 272, so **188 fewer** harmful emails reach the inbox.
- **The cost is small.** Logistic regression marked 14 more legitimate emails for review (148 instead of 134). Precision drops by only 0.12 percentage points.
- **The total error count falls by 43%**, from 406 to 232 wrong predictions out of 16,416 test emails.

### Why the models behave differently

Both models see the same TF-IDF numbers, so the difference comes from how they learn:

- **Naive Bayes** treats every word as independent evidence. It counts how typical each word is for each class and multiplies these values. Emails often contain groups of related words, for example "verify", "account" and "password". Naive Bayes counts this evidence several times, which can push a prediction too far in one direction.
- **Logistic regression** learns all word weights together, so it directly learns which words separate the two classes. Related words share the evidence instead of counting it twice. This helps on borderline emails, where several weak signals have to be combined.

On this data, the extra flexibility mainly helps logistic regression find spam and phishing emails that Naive Bayes classed as legitimate.

Given our cost reasoning, logistic regression is the stronger candidate. We make the final model choice in a later milestone (see Limitations).

### Limitations

- **Selection on the test set.** We compared the two models on the same test set that reports the final score. This is acceptable for one baseline comparison. Before we tune settings or choose the final model, we will add a validation split or cross-validation and keep the test set for one last check.
- **Similar emails across sets.** We only remove exact copies. Emails built from the same template, with only a name or link changed, can still appear in both the training set and the test set. This can make the scores look better than they would be on new emails.
- **Public data instead of company data.** The CSV combines several older public email collections. Phishing styles change over time, and emails in a real company may be different. The scores describe this dataset, not live performance.
- **Fixed threshold.** Both models use the default threshold of 0.5. A different threshold could trade more reviews for fewer missed phishing emails. We have not tested this yet.
- **No time order.** Without timestamps we cannot test whether a model trained on older emails still works on newer ones.
>>>>>>> a6a29023fba16536b99de7152966b631d987e1bb

We plan to add `notebooks/01_data_exploration.ipynb` to show class counts, data quality checks, and a chart based on `results/metrics.csv`.

## Experiment tracking

We will use MLflow to record each training run. Each run will include the model settings, dataset and split information, evaluation scores, and artifacts. This will help us compare the models and find which settings produced each result.

## Model files and FastAPI

We will save the chosen pipeline in `models/`, for example as `phishing_pipeline.joblib`. The file will contain both TF-IDF and the trained classifier so the API can use them together.

The FastAPI service in `app/` will load the saved pipeline when it starts. A user will send email text to `POST /predict`. The API will validate the input, classify the email, and return the predicted class or a clear error.

## Project status

| Milestone                                  | Date      | Status                                                    |
| ------------------------------------------ | --------- | --------------------------------------------------------- |
| Business problem                           | 05 Nov    | Done                                                      |
| Repo, dataset, baseline and improved model | 12–19 Nov | Done (this README, `src/train.py`, `results/metrics.csv`) |
| MLflow tracking                            | 26 Nov    | Next step                                                 |
| Best model selection                       | 03–10 Dec | Planned                                                   |
| Model artifact and input/output schema     | 17 Dec    | Planned                                                   |
| FastAPI deployment                         | 07–14 Jan | Planned                                                   |
| Final presentation                         | 21 Jan    | Planned                                                   |

## Repository layout

<<<<<<< HEAD
| Path | What goes here |
| --- | --- |
| `README.md` | Project description and setup instructions |
| `requirements.txt` | Python packages needed to run training |
| `data/phishing_email.csv` | Original email texts and labels |
| `src/train.py` | Data cleaning, the 80/20 split, training, and model comparison |
| `results/metrics.csv` | Scores written by the training script |
| `models/` | Planned location for the saved pipeline |
| `notebooks/` | Planned Jupyter notebook for data checks and model results |
| `app/` | Planned FastAPI inference service |
=======
| Path                           | What goes here                                                         |
| ------------------------------ | ---------------------------------------------------------------------- |
| `README.md`                    | The project description and setup instructions                         |
| `requirements.txt`             | The Python packages needed to run training                             |
| `data/phishing_email.csv`      | The original email texts and their labels                              |
| `src/train.py`                 | Data cleaning, the 80/20 split, training and model comparison          |
| `results/metrics.csv`          | The scores of both models, written by the training script              |
| `results/baseline_metrics.csv` | The baseline scores, written by `--baseline-only`                      |
| `models/`                      | The saved pipeline for the API, with TF-IDF and the trained classifier |
| `notebooks/`                   | Our Jupyter notebook for data checks and charts of the model results   |
| `app/`                         | The FastAPI service that will use the saved pipeline                   |
>>>>>>> a6a29023fba16536b99de7152966b631d987e1bb

A `.gitkeep` file lets Git track an otherwise empty folder.