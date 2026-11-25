"""Neural text classifiers: a word-level CNN and a fine-tuned DistilBERT.

PyTorch and Hugging Face Transformers are imported only when one of these
models is trained, so the scikit-learn models run without them.

Both models need a validation set to decide when training is good enough.
It is taken from the training emails only, so the test set stays untouched.
"""

import json
import math
import random
import re
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

VALIDATION_SIZE = 0.10  # share of the training emails used for validation
PREDICT_BATCH_SIZE = 64
PROGRESS_EVERY = 200  # print a progress line after this many training batches

# Word-level CNN
CNN_VOCAB_SIZE = 30_000
CNN_MIN_FREQ = 2
CNN_MAX_TOKENS = 300
CNN_EMBEDDING_DIM = 128
CNN_KERNEL_SIZES = (3, 4, 5)
CNN_FILTERS = 100
CNN_DROPOUT = 0.5
CNN_BATCH_SIZE = 64
CNN_LEARNING_RATE = 1e-3
CNN_MAX_EPOCHS = 5
CNN_PATIENCE = 1  # stop after this many epochs without a better validation F1

# Fine-tuned DistilBERT
BERT_CHECKPOINT = "distilbert-base-uncased"
BERT_MAX_TOKENS = 256
BERT_BATCH_SIZE = 16
BERT_LEARNING_RATE = 2e-5
BERT_EPOCHS = 1
BERT_WARMUP_SHARE = 0.10

# Words and numbers become tokens; each other visible character, such as $ or !,
# becomes its own token. The text is already lower case.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")


def _torch():
    import torch

    return torch


def set_seed(seed: int):
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    """Use a GPU when one is available, otherwise the CPU."""
    torch = _torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_validation(texts, labels, seed: int):
    """Hold back a stratified part of the training emails for validation."""
    return train_test_split(
        list(texts),
        np.asarray(labels, dtype=np.int64),
        test_size=VALIDATION_SIZE,
        random_state=seed,
        stratify=np.asarray(labels, dtype=np.int64),
    )


def _clone_state(network):
    return {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}


class WordTokenizer:
    """Split text into tokens and map each known token to a number.

    Id 0 pads short emails; id 1 stands for every token missing from the vocabulary.
    """

    PAD = 0
    UNK = 1

    def __init__(self, vocabulary: dict):
        self.vocabulary = vocabulary

    @classmethod
    def fit(cls, texts, max_size: int, min_freq: int):
        """Learn the most frequent tokens of the given (training) texts."""
        counts = Counter(token for text in texts for token in TOKEN_PATTERN.findall(text))
        common = [
            token for token, count in counts.most_common(max_size - 2) if count >= min_freq
        ]
        return cls({token: index + 2 for index, token in enumerate(common)})

    @property
    def size(self) -> int:
        return len(self.vocabulary) + 2

    def tokenize(self, text: str, max_tokens: int):
        return [match.group() for match in islice(TOKEN_PATTERN.finditer(text), max_tokens)]

    def encode(self, texts, max_tokens: int) -> np.ndarray:
        """Turn texts into a matrix of token ids, cut or padded to max_tokens."""
        ids = np.zeros((len(texts), max_tokens), dtype=np.int32)
        for row, text in enumerate(texts):
            tokens = self.tokenize(text, max_tokens)
            ids[row, : len(tokens)] = [self.vocabulary.get(token, self.UNK) for token in tokens]
        return ids


def _build_text_cnn(vocab_size: int):
    torch = _torch()
    nn = torch.nn

    class TextCNN(nn.Module):
        """Embedding, parallel 1D convolutions over 3, 4 and 5 tokens, max pooling."""

        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, CNN_EMBEDDING_DIM, padding_idx=0)
            self.convolutions = nn.ModuleList(
                nn.Conv1d(CNN_EMBEDDING_DIM, CNN_FILTERS, kernel_size)
                for kernel_size in CNN_KERNEL_SIZES
            )
            self.dropout = nn.Dropout(CNN_DROPOUT)
            self.output = nn.Linear(CNN_FILTERS * len(CNN_KERNEL_SIZES), 1)

        def forward(self, token_ids):
            # (batch, tokens, embedding) -> (batch, embedding, tokens) for Conv1d
            vectors = self.embedding(token_ids).transpose(1, 2)
            pooled = [torch.relu(conv(vectors)).amax(dim=2) for conv in self.convolutions]
            return self.output(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)

    return TextCNN()


class TextCNNClassifier:
    """Word tokenizer plus a convolutional network trained from scratch."""

    def __init__(self, seed: int):
        self.seed = seed

    def fit(self, texts, labels):
        torch = _torch()
        set_seed(self.seed)
        fit_texts, val_texts, fit_labels, val_labels = split_validation(texts, labels, self.seed)

        print("  CNN: building the vocabulary from the training emails")
        self.tokenizer = WordTokenizer.fit(fit_texts, CNN_VOCAB_SIZE, CNN_MIN_FREQ)
        fit_ids = self.tokenizer.encode(fit_texts, CNN_MAX_TOKENS)
        val_ids = self.tokenizer.encode(val_texts, CNN_MAX_TOKENS)

        self.device = pick_device()
        self.network = _build_text_cnn(self.tokenizer.size).to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=CNN_LEARNING_RATE)
        loss_function = torch.nn.BCEWithLogitsLoss()
        shuffler = np.random.default_rng(self.seed)

        best_f1, best_state, epochs_without_gain = -1.0, None, 0
        started = time.perf_counter()
        for epoch in range(1, CNN_MAX_EPOCHS + 1):
            self.network.train()
            order = shuffler.permutation(len(fit_ids))
            for step, start in enumerate(range(0, len(order), CNN_BATCH_SIZE), start=1):
                batch = order[start : start + CNN_BATCH_SIZE]
                token_ids = torch.as_tensor(fit_ids[batch], dtype=torch.long, device=self.device)
                target = torch.as_tensor(fit_labels[batch], dtype=torch.float32, device=self.device)
                optimizer.zero_grad()
                loss = loss_function(self.network(token_ids), target)
                loss.backward()
                optimizer.step()
                if step % PROGRESS_EVERY == 0:
                    print(f"  CNN epoch {epoch}, batch {step}: loss {loss.item():.4f}")

            validation_f1 = f1_score(val_labels, self._predict_ids(val_ids))
            minutes = (time.perf_counter() - started) / 60
            print(f"  CNN epoch {epoch}: validation F1 {validation_f1:.4f} ({minutes:.1f} min)")
            if validation_f1 > best_f1:
                best_f1, best_state, epochs_without_gain = validation_f1, _clone_state(self.network), 0
                self.best_epoch = epoch
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= CNN_PATIENCE:
                    break

        self.network.load_state_dict(best_state)
        self.validation_f1 = best_f1
        return self

    def _predict_ids(self, ids: np.ndarray) -> np.ndarray:
        torch = _torch()
        self.network.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(ids), PREDICT_BATCH_SIZE):
                batch = torch.as_tensor(
                    ids[start : start + PREDICT_BATCH_SIZE], dtype=torch.long, device=self.device
                )
                # A logit above 0 means a probability above 0.5 for spam or phishing.
                predictions.append((self.network(batch) > 0).long().cpu().numpy())
        return np.concatenate(predictions)

    def predict(self, texts) -> np.ndarray:
        return self._predict_ids(self.tokenizer.encode(list(texts), CNN_MAX_TOKENS))

    def params(self) -> dict:
        return {
            "features": "own word tokenizer + learned embeddings",
            "classifier": "TextCNN",
            "validation_size": VALIDATION_SIZE,
            "tokenizer.vocab_size": self.tokenizer.size,
            "tokenizer.min_freq": CNN_MIN_FREQ,
            "tokenizer.max_tokens": CNN_MAX_TOKENS,
            "cnn.embedding_dim": CNN_EMBEDDING_DIM,
            "cnn.kernel_sizes": ",".join(map(str, CNN_KERNEL_SIZES)),
            "cnn.filters": CNN_FILTERS,
            "cnn.dropout": CNN_DROPOUT,
            "cnn.batch_size": CNN_BATCH_SIZE,
            "cnn.learning_rate": CNN_LEARNING_RATE,
            "cnn.max_epochs": CNN_MAX_EPOCHS,
            "cnn.best_epoch": self.best_epoch,
            "device": self.device.type,
        }

    def extra_metrics(self) -> dict:
        return {"validation_f1": float(self.validation_f1)}

    def save(self, folder: Path):
        torch = _torch()
        torch.save(self.network.state_dict(), folder / "text_cnn.pt")
        (folder / "vocabulary.json").write_text(json.dumps(self.tokenizer.vocabulary), "utf-8")
        (folder / "config.json").write_text(json.dumps(self.params(), indent=2), "utf-8")

    def log_to_mlflow(self, mlflow):
        with TemporaryDirectory() as folder:
            self.save(Path(folder))
            mlflow.log_artifacts(folder, artifact_path="model")


class DistilBertClassifier:
    """Pretrained DistilBERT language model, fine-tuned on the training emails."""

    def __init__(self, seed: int, train_limit: int | None = None, max_tokens: int = BERT_MAX_TOKENS):
        self.seed = seed
        self.train_limit = train_limit
        self.max_tokens = max_tokens

    def _encode(self, texts):
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_tokens,
            padding=True,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _autocast(self):
        """Half precision on an NVIDIA GPU: about half the memory and faster."""
        torch = _torch()
        return torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.mixed_precision
        )

    def fit(self, texts, labels):
        torch = _torch()
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )

        set_seed(self.seed)
        texts = list(texts)
        labels = np.asarray(labels, dtype=np.int64)
        if self.train_limit and self.train_limit < len(texts):
            # Optional smaller training sample for computers without a GPU.
            texts, _, labels, _ = train_test_split(
                texts,
                labels,
                train_size=self.train_limit,
                random_state=self.seed,
                stratify=labels,
            )
        self.train_emails = len(texts)
        fit_texts, val_texts, fit_labels, val_labels = split_validation(texts, labels, self.seed)

        self.device = pick_device()
        self.mixed_precision = self.device.type == "cuda"
        print(f"  DistilBERT: loading {BERT_CHECKPOINT} (first run downloads about 270 MB)")
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT)
        self.network = AutoModelForSequenceClassification.from_pretrained(
            BERT_CHECKPOINT, num_labels=2
        ).to(self.device)

        steps_per_epoch = math.ceil(len(fit_texts) / BERT_BATCH_SIZE)
        total_steps = steps_per_epoch * BERT_EPOCHS
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=BERT_LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * BERT_WARMUP_SHARE), total_steps
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        shuffler = np.random.default_rng(self.seed)
        print(
            f"  DistilBERT: {len(fit_texts):,} training emails, {total_steps:,} steps on "
            f"{self.device.type}" + (" with mixed precision" if self.mixed_precision else "")
        )

        best_f1, best_state = -1.0, None
        started = time.perf_counter()
        for epoch in range(1, BERT_EPOCHS + 1):
            self.network.train()
            order = shuffler.permutation(len(fit_texts))
            for step, start in enumerate(range(0, len(order), BERT_BATCH_SIZE), start=1):
                batch = order[start : start + BERT_BATCH_SIZE]
                inputs = self._encode([fit_texts[index] for index in batch])
                target = torch.as_tensor(fit_labels[batch], dtype=torch.long, device=self.device)
                optimizer.zero_grad()
                with self._autocast():
                    loss = self.network(**inputs, labels=target).loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                if step % PROGRESS_EVERY == 0:
                    minutes = (time.perf_counter() - started) / 60
                    print(
                        f"  DistilBERT epoch {epoch}, step {step}/{steps_per_epoch}: "
                        f"loss {loss.item():.4f} ({minutes:.1f} min)"
                    )

            validation_f1 = f1_score(val_labels, self.predict(val_texts))
            print(f"  DistilBERT epoch {epoch}: validation F1 {validation_f1:.4f}")
            if validation_f1 > best_f1:
                best_f1, self.best_epoch = validation_f1, epoch
                if BERT_EPOCHS > 1:
                    best_state = _clone_state(self.network)

        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.validation_f1 = best_f1
        return self

    def predict(self, texts) -> np.ndarray:
        torch = _torch()
        texts = list(texts)
        # Emails of similar length share a batch, so little padding is computed.
        # The predictions are put back into the original order afterwards.
        order = np.argsort([len(text) for text in texts], kind="stable")
        predictions = np.empty(len(texts), dtype=np.int64)
        self.network.eval()
        started = time.perf_counter()
        with torch.no_grad(), self._autocast():
            for batch, start in enumerate(range(0, len(texts), PREDICT_BATCH_SIZE), start=1):
                rows = order[start : start + PREDICT_BATCH_SIZE]
                inputs = self._encode([texts[row] for row in rows])
                predictions[rows] = self.network(**inputs).logits.argmax(dim=1).cpu().numpy()
                if batch % 50 == 0:
                    done = min(start + PREDICT_BATCH_SIZE, len(texts))
                    minutes = (time.perf_counter() - started) / 60
                    print(f"  DistilBERT predicting: {done:,}/{len(texts):,} emails ({minutes:.1f} min)")
        return predictions

    def params(self) -> dict:
        return {
            "features": "DistilBERT WordPiece tokenizer",
            "classifier": "DistilBertForSequenceClassification",
            "validation_size": VALIDATION_SIZE,
            "bert.checkpoint": BERT_CHECKPOINT,
            "bert.max_tokens": self.max_tokens,
            "bert.batch_size": BERT_BATCH_SIZE,
            "bert.learning_rate": BERT_LEARNING_RATE,
            "bert.epochs": BERT_EPOCHS,
            "bert.warmup_share": BERT_WARMUP_SHARE,
            "bert.best_epoch": self.best_epoch,
            "bert.train_emails": self.train_emails,
            "bert.mixed_precision": self.mixed_precision,
            "device": self.device.type,
        }

    def extra_metrics(self) -> dict:
        return {"validation_f1": float(self.validation_f1)}

    def save(self, folder: Path):
        self.network.save_pretrained(folder)
        self.tokenizer.save_pretrained(folder)

    def log_to_mlflow(self, mlflow):
        with TemporaryDirectory() as folder:
            self.save(Path(folder))
            mlflow.log_artifacts(folder, artifact_path="model")