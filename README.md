# Spam and Phishing Email Detection

This AIOps project reads email text and predicts one of two classes. Label `0` means legitimate email. Label `1` means spam or phishing. Employees and IT teams can use the result to decide which messages need a closer check.

## Dataset

We use `phishing_email.csv` from the *Phishing Email Dataset* on Kaggle, published by Naser Abdullah Alam with Amith Khandakar. The file has 82,486 rows. `text_combined` contains the email text, and `label` contains the class. [Dataset DOI](https://doi.org/10.34740/kaggle/ds/5074342) · [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/)

| Label | Meaning | Rows in the original CSV | Suggested action |
| --- | --- | ---: | --- |
| `0` | Legitimate email | 39,595 | Deliver normally |
| `1` | Spam or phishing | 42,891 | Flag for review |

The labels are already in the dataset. Spam and phishing use the same label.

The data card asks users to cite Al-Subaiey et al. (2024), *Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection*. [Paper](https://arxiv.org/abs/2405.11619)

## Setup and training

Install Python 3.12. Download `phishing_email.csv` and put it in `data/phishing_email.csv`. If the download is a ZIP, extract the CSV first. [Kaggle download](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=phishing_email.csv)

Open PowerShell in the main project folder. These commands create a Python environment, install the required packages, and run training.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/train.py
```

If the environment is already active and the packages are installed, run `python src/train.py`.

## What `train.py` does

1. Read the CSV. Normalize whitespace and letter case. Remove rows with missing text or labels, repeated texts, and texts with conflicting labels.
2. Split the cleaned emails into 80% training and 20% testing. Stratification keeps the class proportions similar in both sets. Seed `42` makes the split repeatable with the same input data.
3. Train both models on the same training emails. Each model learns TF-IDF and the classifier from the training set.
4. Ask both models to predict the classes of the test emails using only their text. Compare the predictions with the known labels.
5. Print the results and save the scores in `results/metrics.csv`.

Both models use the same test set, so we can compare their results directly.

## The two models

| Model | Purpose |
| --- | --- |
| TF-IDF and multinomial Naive Bayes | Our baseline for the comparison |
| TF-IDF and logistic regression | The second model we compare with the baseline |

TF-IDF stands for Term Frequency and Inverse Document Frequency. It turns email text into numbers based on how often words appear. Words that appear in many emails usually get less weight. The classifier uses these numbers to find patterns.

A pipeline joins TF-IDF and a classifier into one object. The training script fits both pipelines and saves their evaluation scores in `results/metrics.csv`.

## Results

The CSV contained 82,486 rows. Cleaning removed one row with missing text or a missing label and 408 duplicate texts, leaving 82,077 emails.

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

We plan to add `notebooks/01_data_exploration.ipynb` to show class counts, data quality checks, and a chart based on `results/metrics.csv`.

## Experiment tracking

We will use MLflow to record each training run. Each run will include the model settings, dataset and split information, evaluation scores, and artifacts. This will help us compare the models and find which settings produced each result.

## Model files and FastAPI

We will save the chosen pipeline in `models/`, for example as `phishing_pipeline.joblib`. The file will contain both TF-IDF and the trained classifier so the API can use them together.

The FastAPI service in `app/` will load the saved pipeline when it starts. A user will send email text to `POST /predict`. The API will validate the input, classify the email, and return the predicted class or a clear error.

## Repository layout

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

A `.gitkeep` file lets Git track an otherwise empty folder.