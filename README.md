# Spam and Phishing Email Detection

This AIOps project classifies email text as legitimate (0) or spam/phishing (1). IT security teams can use the result to decide which emails to deliver and which to flag for review. We compare four models on the same test split, record every run in MLflow and serve the chosen model through a FastAPI service.

| Key    | Model                            | Idea                                                                      |
| ------ | -------------------------------- | ------------------------------------------------------------------------- |
| `nb`   | TF-IDF + multinomial Naive Bayes | Baseline: word statistics per class                                       |
| `lr`   | TF-IDF + logistic regression     | One learned weight per word                                               |
| `cnn`  | Own word tokenizer + CNN         | Learns word vectors and short word patterns such as “verify your account” |
| `bert` | Fine-tuned DistilBERT            | Small pretrained language model, fine-tuned on our training emails        |

The first three models learn only from our training emails. DistilBERT starts from a pretrained language model.

- A code-free explanation of the four algorithms with simple diagrams is in [docs/algorithms.md](docs/algorithms.md).
- The notebook [notebooks/01_phishing_detection.ipynb](notebooks/01_phishing_detection.ipynb) walks through the data, the models and the results in VS Code.
- The notebook [notebooks/01_phishing_detection_colab.ipynb](notebooks/01_phishing_detection_colab.ipynb) does the same in Google Colab without any local setup (see [Run in Google Colab](#run-in-google-colab)).

## Dataset

`src/train.py` reads `data/phishing_email.csv`. The `text_combined` column contains the email text, and `label` contains the class. On another PC, place the same CSV at `data/phishing_email.csv` before running the project. [Kaggle dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) · Dataset DOI · CC BY-SA 4.0 license

| Label | Meaning          | Original rows |
| ----- | ---------------- | ------------- |
| 0     | Legitimate email | 39,595        |
| 1     | Spam or phishing | 42,891        |
|       | **Total**        | **82,486**    |

The script normalizes whitespace and letter case, removes rows with empty text or labels, and removes exact duplicates. If the same text has conflicting labels, it removes all copies. The cleaned data is split into 80% training and 20% test data, stratified by label with seed 42.

The CNN and DistilBERT also need validation data to decide when to stop training. They take a stratified 10% of the **training** emails for this. The test set is never used during training.

## Windows setup

Use Python 3.12 and run the commands from the project folder in PowerShell.

### First setup on a PC

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` contains these packages:

```text
pandas
scikit-learn
mlflow
joblib
torch
transformers
matplotlib
ipykernel
fastapi
uvicorn
```

`torch` and `transformers` are needed for the CNN and DistilBERT, `matplotlib` and `ipykernel` for the notebook, and `fastapi` and `uvicorn` for the API. The first DistilBERT run downloads the pretrained model (about 270 MB) from Hugging Face.

### Use an NVIDIA GPU

On Windows, `pip install torch` installs a CPU-only build. With an NVIDIA GPU, replace it with the CUDA build (check the CUDA version with `nvidia-smi` and the matching command on pytorch.org):

```powershell
python -m pip uninstall -y torch
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The CNN and DistilBERT then use the GPU automatically; DistilBERT also uses mixed precision, which about halves its memory use. The training output says `on cuda`.

### Command Prompt instead of PowerShell

The commands in this README are for PowerShell. In the Command Prompt (cmd), activate the environment with `.venv\Scripts\activate.bat` and set variables with `set NAME=value` instead of `$env:NAME = "value"`.

### Open a new terminal

Activate the existing environment in each new PowerShell window.

```powershell
.\.venv\Scripts\Activate.ps1
```

### Check the dataset and split

Use this command as a quick check after setup.

```powershell
python src/train.py --split-only
```

The terminal should show 82,077 emails after cleaning, with 65,661 for training and 16,416 for testing.

To see the available options, run `python src/train.py --help`.

## Train the models

Before training, set the location of the local MLflow database in the current PowerShell window.

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow_local.db"
```

Each PC uses its own database file. A database copied from another computer still points to that computer’s artifact folder; the script stops with an error before training if that folder cannot be used. Keep `mlflow*.db` and `mlruns/` in `.gitignore`.

### Choose the models

| Command                                | What it trains                                                         |
| -------------------------------------- | ---------------------------------------------------------------------- |
| `python src/train.py`                  | All four models                                                        |
| `python src/train.py --models nb lr`   | Naive Bayes and logistic regression                                    |
| `python src/train.py --models cnn`     | Only the CNN                                                           |
| `python src/train.py --models bert`    | Only DistilBERT                                                        |
| `python src/train.py --models cascade` | Optional: logistic regression first, DistilBERT only for unsure emails |
| `python src/train.py --baseline-only`  | Only Naive Bayes, written to `results/baseline_metrics.csv`            |

Each trained model creates one MLflow run. `results/metrics.csv` keeps one row per model: a new run replaces that model's row and keeps the others. So you can train the slow models separately.

To display the comparison CSV in PowerShell, run this command.

```powershell
Import-Csv .\results\metrics.csv | Format-Table -AutoSize
```

### Runtime

Naive Bayes and logistic regression train in seconds. The CNN needs about 7 minutes on a laptop CPU and about a minute on a GPU. DistilBERT needs a GPU: on an RTX 3050 laptop GPU it trains in about 7 minutes, while on a laptop CPU only 9,000 training emails already took about an hour. `train_seconds` in the results shows the real time.

Our reference run for DistilBERT:

```powershell
python src/train.py --models bert --bert-max-tokens 128
```

To make DistilBERT faster, `--bert-max-tokens 128` reads only the first 128 tokens of each email instead of 256; this about halves the training time but the model sees less of each email.

Without a GPU, `--bert-train-limit N` fine-tunes DistilBERT on a stratified sample of N training emails. The test set stays the same, but the model sees less data, so the comparison is not fair to it.

### Optional: faster email checks with a cascade

For a mail server, the time per email matters most. Every run therefore also records `emails_per_second` and `ms_per_email` for the test set.

The cascade is implemented but not part of our reference results yet. It combines speed and quality: logistic regression checks every email in a fraction of a millisecond. Only the emails it is unsure about, with a spam probability between 10% and 90% (`CASCADE_LOW` and `CASCADE_HIGH` in `src/train.py`), go to DistilBERT. The run records which share of the emails went to DistilBERT as `share_to_distilbert`.

```powershell
python src/train.py --models cascade --bert-max-tokens 128
```

## Classify single emails instantly

Every training run also saves the fitted model to `models/<key>/` (for example `models/bert/`). A saved model is loaded **once**; after that, each email only needs a prediction, which takes milliseconds. Each email is cleaned exactly like the training data (whitespace and lower case).

Quick check in the terminal:

```powershell
python src/serving.py bert "URGENT: verify your account password here"
```

### FastAPI service

The folder `app/` contains `main.py` and an empty `__init__.py`. Start the service from the project folder:

```powershell
python -m uvicorn app.main:app --port 8000
```

The terminal shows which model was loaded and whether it runs on the GPU (`cuda`) or the CPU.

Open <http://127.0.0.1:8000/docs>, choose **POST /predict**, click **Try it out**, enter an email text and click **Execute**. The answer contains the label (0 legitimate, 1 spam or phishing), the spam probability, the model, the device and the prediction time in milliseconds. `GET /health` shows which model is loaded and on which device. An empty text returns an error (status 422).

The service uses DistilBERT by default, because it has the best precision and F1. To use another saved model, set `PHISHING_MODEL` before starting it, for example `$env:PHISHING_MODEL = "lr"` in PowerShell or `set PHISHING_MODEL=lr` in cmd. Possible values: `nb`, `lr`, `cnn`, `bert`, `cascade`. The model must be trained first, so that it exists in `models/`.

## Run in Google Colab

The Colab notebook needs no local setup:

1. Open <https://colab.research.google.com>, choose **File → Upload notebook** and select `notebooks/01_phishing_detection_colab.ipynb`.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Choose **Runtime → Run all**.

The first cells install MLflow, write the project code and the diagrams, and download the dataset from Kaggle. If the download does not work, an upload button appears for `phishing_email.csv`. All four models then train on the Colab GPU (about 10–15 minutes), and `results/metrics.csv` is downloaded at the end.

## View the runs in MLflow

Open PowerShell in the project folder with `.venv` active. Use this command after training or to reopen runs already saved in `mlflow_local.db`.

```powershell
mlflow server --backend-store-uri "sqlite:///mlflow_local.db" --host 127.0.0.1 --port 5000
```

Keep the terminal open. In the browser, open <http://127.0.0.1:5000>, select **Spam and Phishing Email Detection** and click **Training runs**.

Open a run to see its parameters and metrics. Under **Artifacts**, open `confusion_matrix.json` to see the classification counts. Naive Bayes and logistic regression store their scikit-learn pipeline as a model. The CNN and DistilBERT store their weights and tokenizer under `model/`, and their runs also show `validation_f1`. Press Ctrl+C in PowerShell to stop the server.

## The four models

All models predict label 1 (spam or phishing) when its probability is above 0.5.

### Naive Bayes and logistic regression

Both use TF-IDF with `min_df=2` and `max_features=30000`. Each pipeline learns its vocabulary and word weights from the training emails. Multinomial Naive Bayes is the baseline. Logistic regression uses `solver="liblinear"`, `max_iter=1000`, and `random_state=42`.

### CNN with a word tokenizer

| Setting      | Value                                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| Tokenizer    | Words, numbers and single symbols such as `$` or `!`; vocabulary of the 30,000 most frequent training tokens (at least 2 occurrences) |
| Email length | First 300 tokens                                                                                                                      |
| Network      | 128-dim embeddings, 100 filters each over 3, 4 and 5 tokens, max pooling, dropout 0.5                                                 |
| Training     | Adam, learning rate 0.001, batch 64, up to 5 epochs, stops when validation F1 stops improving                                         |

### Fine-tuned DistilBERT

| Setting          | Value                                                                              |
| ---------------- | ---------------------------------------------------------------------------------- |
| Pretrained model | `distilbert-base-uncased` with its WordPiece tokenizer                             |
| Email length     | First 256 tokens                                                                   |
| Training         | AdamW, learning rate 2e-5, batch 16, 1 epoch, 10% warmup, mixed precision on a GPU |

The CNN and DistilBERT settings are constants at the top of `src/deep_models.py`.

## Results

Recall is our main metric. It measures the share of spam or phishing emails the model catches. Precision measures the share of flagged emails that really were spam or phishing. F1 combines precision and recall.

| Model                               | Precision  | Recall     | F1         | Legitimate emails flagged by mistake | Spam or phishing emails missed |
| ----------------------------------- | ---------- | ---------- | ---------- | ------------------------------------ | ------------------------------ |
| Multinomial Naive Bayes             | 98.41%     | 96.81%     | 97.61%     | 134                                  | 273                            |
| Logistic regression                 | 98.29%     | 99.02%     | 98.65%     | 148                                  | 84                             |
| CNN (word tokenizer)                | 98.56%     | **99.14%** | 98.85%     | 124                                  | **74**                         |
| DistilBERT (fine-tuned, 128 tokens) | **99.13%** | 98.96%     | **99.05%** | **74**                               | 89                             |

The test set has 8,569 spam or phishing emails and 7,847 legitimate emails. On it, logistic regression missed 189 fewer spam or phishing emails than Naive Bayes and flagged 14 more legitimate emails for review. The CNN improves both error types again: compared with logistic regression it missed 10 fewer spam or phishing emails and flagged 24 fewer legitimate emails.

| Model                               | Training time                               | Prediction time for 16,416 test emails |
| ----------------------------------- | ------------------------------------------- | -------------------------------------- |
| Multinomial Naive Bayes             | 7 s                                         | 1.3 s                                  |
| Logistic regression                 | 9 s                                         | 1.4 s                                  |
| CNN (word tokenizer)                | 7.0 min (4 epochs on CPU)                   | 7.0 s                                  |
| DistilBERT (fine-tuned, 128 tokens) | 7.2 min (1 epoch on an RTX 3050 laptop GPU) | 20.8 s                                 |

DistilBERT makes the fewest errors overall (163 instead of 198 for the CNN and 232 for logistic regression) and flags the fewest legitimate emails (74). The CNN still misses the fewest spam or phishing emails (74 instead of 89), so it has the highest recall, our main metric. Both neural models train about 50 times longer than logistic regression, and DistilBERT needs a GPU and predicts about 15 times slower. Logistic regression is the easiest to explain, because each word has one weight.

### Which model is best?

- **Naive Bayes** is the weakest, because it looks at every word alone.
- **Logistic regression** is much better, because it learns the weights of all words together. It is the fastest and easiest to explain.
- **The CNN** finds the most spam (best recall), because it also sees the order of words in short phrases.
- **DistilBERT** makes the fewest mistakes overall and has the fewest false alarms (best precision and F1), because it is pretrained on English text and reads every word in its context. It needs a GPU.

The best choice depends on the goal: the CNN if catching as much phishing as possible matters most, DistilBERT for the fewest mistakes and false alarms, logistic regression if speed and explainability matter most. The API uses DistilBERT by default.

### Limitations

These results come from one split of a public dataset. Similar email templates can appear in both sets after exact duplicate removal. A company inbox can contain a different mix of emails. The models are still compared on the test set; the final model choice will also weigh runtime and explainability.

## Project status

| Milestone                                            | Date      | Status                                 |
| ---------------------------------------------------- | --------- | -------------------------------------- |
| Business problem                                     | 05 Nov    | Done                                   |
| Repo, dataset, baseline and improved model           | 12–19 Nov | Done                                   |
| MLflow tracking and further models (CNN, DistilBERT) | 26 Nov    | Done                                   |
| Best model selection                                 | 03–10 Dec | Next step                              |
| Model artifact and input/output schema               | 17 Dec    | Prepared (`models/`, `src/serving.py`) |
| FastAPI deployment                                   | 07–14 Jan | Prepared (`app/main.py`)               |
| Final presentation                                   | 21 Jan    | Planned                                |

## Project files

| Path                                          | Purpose                                                             |
| --------------------------------------------- | ------------------------------------------------------------------- |
| `src/train.py`                                | Data cleaning, splitting, training, evaluation, and MLflow tracking |
| `src/deep_models.py`                          | Word tokenizer, CNN and DistilBERT fine-tuning                      |
| `requirements.txt`                            | Python dependencies                                                 |
| `data/phishing_email.csv`                     | Email text and labels                                               |
| `results/metrics.csv`                         | One row per trained model                                           |
| `results/baseline_metrics.csv`                | Results generated by the baseline-only command                      |
| `models/<key>/`                               | Saved model of each trained model, loaded by the API                |
| `src/serving.py`                              | Loads a saved model and classifies single emails                    |
| `app/main.py` and `app/__init__.py`           | FastAPI service with `POST /predict` and `GET /health`              |
| `notebooks/01_phishing_detection.ipynb`       | Walk-through of data, models, results and MLflow runs (VS Code)     |
| `notebooks/01_phishing_detection_colab.ipynb` | The same walk-through for Google Colab, self-contained              |
| `docs/algorithms.md` and `docs/img/`          | Code-free explanation of the four algorithms with simple diagrams   |
| `mlflow_local.db` and `mlruns/`               | Local MLflow database and artifacts                                 |

The dataset card asks users to cite Al-Subaiey et al. (2024), _Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection_. Paper
