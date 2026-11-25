"""Tests for the FastAPI service with a small fake model, so no training is needed.

Run from the project folder:  python -m pytest tests
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main


class FakeModel:
    """Says spam when the email contains 'verify'."""

    def predict_proba(self, texts):
        spam = np.array([0.97 if "verify" in text else 0.02 for text in texts])
        return np.column_stack([1 - spam, spam])


class BrokenModel:
    def predict_proba(self, texts):
        raise RuntimeError("model crashed")


def client_with(monkeypatch, model=None, error=None):
    def fake_load(key):
        if error:
            raise FileNotFoundError(error)
        return "Fake model", model

    monkeypatch.setattr(main, "load_model", fake_load)
    return TestClient(main.app)


@pytest.fixture
def client(monkeypatch):
    with client_with(monkeypatch, FakeModel()) as test_client:
        yield test_client


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_spam(client):
    response = client.post("/predict", json={"text": "Please VERIFY your account"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == 1
    assert body["label_name"] == "spam or phishing"
    assert 0 <= body["spam_probability"] <= 1


def test_predict_legitimate(client):
    response = client.post("/predict", json={"text": "Meeting moved to Thursday"})
    assert response.status_code == 200
    assert response.json()["label"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},                                   # empty text
        {"text": "   \n  "},                            # only whitespace
        {},                                             # text missing
        {"text": 123},                                  # not a string
        {"text": "hello", "sender": "a@b.com"},         # unknown field
        {"text": "x" * (main.MAX_CHARACTERS + 1)},      # too long
    ],
)
def test_invalid_input_returns_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_stats_count_requests(client):
    client.post("/predict", json={"text": "verify now"})
    client.post("/predict", json={"text": "lunch tomorrow"})
    stats = client.get("/stats").json()
    assert stats["requests"] == 2
    assert stats["spam_share"] == 0.5


def test_missing_model_gives_503(monkeypatch):
    with client_with(monkeypatch, error="No saved model") as test_client:
        assert test_client.get("/health").status_code == 503
        response = test_client.post("/predict", json={"text": "hello"})
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"]


def test_model_error_gives_500(monkeypatch):
    with client_with(monkeypatch, BrokenModel()) as test_client:
        response = test_client.post("/predict", json={"text": "hello"})
        assert response.status_code == 500
        assert test_client.get("/stats").json()["failures"] == 1