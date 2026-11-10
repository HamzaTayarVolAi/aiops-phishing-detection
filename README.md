# Spam and Phishing Email Detection

This AIOps project reads email text and predicts one of two classes. Label `0` means legitimate email. Label `1` means spam or phishing. Employees and IT teams can use the result to choose which messages need a closer check.

## Dataset

We use `phishing_email.csv` from the *Phishing Email Dataset* on Kaggle, published by Naser Abdullah Alam with Amith Khandakar. The file has 82,486 rows. `text_combined` contains the email text, and `label` contains the correct class. [Dataset DOI](https://doi.org/10.34740/kaggle/ds/5074342) · [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/)

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

## What train.py does

1. Read the original CSV. Clean spaces and letter case. Remove empty texts, repeated texts, and texts with different labels for the same message.
2. Put 80% of the emails in the training set and 20% in the test set. Both sets contain legitimate and spam or phishing emails. Seed `42` makes the split repeatable with the same input data.
3. Train two models on the same training emails. Each model learns from the email text and its correct label.
4. Ask both models to predict the classes of the test emails using only their text. Compare the predictions with the known labels.
5. Print the results and save the scores in `results/metrics.csv`.

The test emails help us check how well the models work on examples kept out of training. Both models use the same split, so we can compare their results fairly.

## The two models

| Model | Purpose |
| --- | --- |
| TF-IDF and multinomial Naive Bayes | Our baseline, which gives us a first result to compare with |
| TF-IDF and logistic regression | A second model that we compare with the baseline |

TF-IDF stands for Term Frequency and Inverse Document Frequency. It gives words a score and turns email text into numbers. Words that appear in many emails usually get less weight. These numbers help the classifier find patterns.

A pipeline joins TF-IDF and a classifier into one object. During training, each pipeline learns its word rules from the training emails.

The script creates and trains both pipelines in the computer's memory while it runs. The scores in `results/metrics.csv` show how they performed. A saved pipeline file in `models/` holds what the chosen model learned, so the API can use it after the training script ends.

## Results

Our project will include `notebooks/01_data_exploration.ipynb`. This Jupyter notebook will show the class counts and checks for missing or repeated texts. It will also read `results/metrics.csv` and show the model scores in a chart. A notebook puts code, notes and charts together, which helps us explain the data and the results.

Precision tells us how many emails marked as suspicious really belong to label `1`. Recall tells us how many of the suspicious emails the model found. F1 combines precision and recall into one score.

A false positive is a legitimate email marked as suspicious. A false negative is a spam or phishing email marked as legitimate.

The following reference run used 65,661 training emails and 16,416 test emails after cleaning.

| Model with TF-IDF | Precision | Recall | F1 | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multinomial Naive Bayes | 98.41% | 96.83% | 97.61% | 134 | 272 |
| Logistic regression | 98.29% | 99.02% | 98.65% | 148 | 84 |

On this split, logistic regression missed 188 fewer spam or phishing emails. It also marked 14 more legitimate emails for review.

These scores describe this public dataset. Similar email templates may appear in both sets, and emails in a real company may be different.

## Model files and FastAPI

The current training script creates the model comparison. The model export and API steps below describe the design for using the chosen pipeline.

We will save the chosen pipeline in `models/`, for example as `phishing_pipeline.joblib`. Joblib is a Python library that saves and loads trained models. The file will contain TF-IDF and the classifier together, so both steps are ready to use again.

The FastAPI service in `app/` will load this file when it starts. Loading means reading the file from the computer where the API runs. If the API runs on another computer, we first copy the model file there.

A user will send email text to `POST /predict`. The API will pass the text to the loaded pipeline and return the predicted class. The API will check the input and give a clear result or error.

## Experiment tracking

We will use MLflow to keep a record of each training run. Each record will contain the model settings, scores and saved files. This will help us compare runs and find which settings produced a result.

## Repository layout

| Path | What goes here |
| --- | --- |
| `README.md` | The project description and setup instructions |
| `requirements.txt` | The Python packages needed to run training |
| `data/phishing_email.csv` | The original email texts and their labels |
| `src/train.py` | Data cleaning, the 80/20 split, training and model comparison |
| `results/metrics.csv` | The scores written by the training script |
| `models/` | The saved pipeline for the API, with TF-IDF and the trained classifier |
| `notebooks/` | Our Jupyter notebook for data checks and charts of the model results |
| `app/` | The FastAPI service that will use the saved pipeline |

A `.gitkeep` file keeps an empty folder in Git. The folders will hold their model files, notebook or API code as we build each part.
