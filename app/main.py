"""FastAPI service that classifies one email at a time.

Start it from the project folder:
    python -m uvicorn app.main:app --port 8000
Then open http://127.0.0.1:8000/docs.

Choose the model with the environment variable PHISHING_MODEL (nb, lr, cnn, bert, cascade);
the default is bert. The model is loaded once at start-up, so each request only predicts.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from serving import classify, load_model, model_device  # noqa: E402

MODEL_KEY = os.environ.get("PHISHING_MODEL", "bert")
METRICS_PATH = ROOT / "results" / "metrics.csv"
MAX_CHARACTERS = 200_000

logger = logging.getLogger("phishing_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

state = {}


def reset_state():
    state.clear()
    state.update(
        key=MODEL_KEY, name=None, model=None, device=None, error=None,
        started=time.time(), requests=0, spam=0, failures=0, total_ms=0.0,
    )


reset_state()


class Email(BaseModel):
    """Input: one email. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CHARACTERS,
        description="Subject and body of the email as plain text.",
        examples=["URGENT: your account was suspended. Verify your password here: http://secure-login.example.com"],
    )


class Prediction(BaseModel):
    label: int = Field(description="0 = legitimate, 1 = spam or phishing")
    label_name: str = Field(description="legitimate or spam or phishing")
    spam_probability: float = Field(description="Probability of spam or phishing, from 0 to 1")
    model: str
    device: str = Field(description="cuda = NVIDIA GPU, cpu = processor")
    milliseconds: float = Field(description="Prediction time for this email")


@asynccontextmanager
async def lifespan(_: FastAPI):
    reset_state()
    try:
        state["name"], state["model"] = load_model(MODEL_KEY)
        state["device"] = model_device(state["model"])
        logger.info("Loaded %s on %s", state["name"], state["device"])
    except (OSError, ValueError, ImportError) as error:
        # The service still starts, so /health can explain what is missing.
        state["error"] = str(error)
        logger.error("Model '%s' could not be loaded: %s", MODEL_KEY, error)
    if state["model"] is not None:
        try:
            classify(state["model"], "warm-up email")  # the first prediction is slower on a GPU
        except Exception:  # noqa: BLE001 - a failing model is reported per request instead
            logger.exception("Warm-up prediction failed")
    yield
    state.clear()


app = FastAPI(
    title="Spam and Phishing Email Detection",
    description="Classifies one email as legitimate (0) or spam/phishing (1).",
    version="1.0",
    lifespan=lifespan,
)


def require_model():
    if state.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{MODEL_KEY}' is not loaded: {state.get('error')}",
        )


@app.get("/health")
def health():
    """Is the service ready? 200 = ready, 503 = model missing."""
    if state.get("model") is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model": MODEL_KEY, "error": state.get("error")},
        )
    return {"status": "ok", "model": state["name"], "device": state["device"]}


@app.post("/predict", response_model=Prediction)
def predict(email: Email):
    """Classify one email."""
    require_model()
    try:
        result = classify(state["model"], email.text)
    except ValueError as error:  # for example an email that is only whitespace
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:  # an unexpected problem inside the model
        state["failures"] += 1
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. See the server log.") from error

    state["requests"] += 1
    state["spam"] += result["label"]
    state["total_ms"] += result["milliseconds"]
    logger.info(
        "predict label=%s probability=%.4f ms=%.2f",
        result["label"], result["spam_probability"], result["milliseconds"],
    )
    return Prediction(model=state["name"], device=state["device"], **result)


@app.get("/model")
def model_info():
    """The loaded model and its test results from results/metrics.csv."""
    require_model()
    info = {"key": MODEL_KEY, "name": state["name"], "device": state["device"]}
    if METRICS_PATH.is_file():
        metrics = pd.read_csv(METRICS_PATH).set_index("model")
        if state["name"] in metrics.index:
            row = metrics.loc[state["name"]]
            info["test_results"] = {
                column: round(float(row[column]), 4)
                for column in ["precision", "recall", "f1", "false_positives", "false_negatives"]
                if column in row and pd.notna(row[column])
            }
    return info


@app.get("/stats")
def stats():
    """Simple monitoring: requests since start-up."""
    answered = state.get("requests", 0)
    return {
        "model": state.get("name") or MODEL_KEY,
        "uptime_seconds": round(time.time() - state.get("started", time.time())),
        "requests": answered,
        "spam_share": round(state.get("spam", 0) / answered, 4) if answered else None,
        "average_milliseconds": round(state.get("total_ms", 0.0) / answered, 2) if answered else None,
        "failures": state.get("failures", 0),
    }