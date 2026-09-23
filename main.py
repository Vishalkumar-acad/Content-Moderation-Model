"""
Content Moderation Model — FastAPI inference server (for Render.com).

Loads the TF-IDF vectorizer (tfidf_vectorizer.pkl) and the ONNX model
(content_moderation_gpu.onnx, weights in content_moderation_gpu.onnx.data)
and exposes:

  GET  /         -> service health + model status
  POST /moderate -> {"text": "..."} -> toxicity score

The server starts even if the model is broken (e.g. truncated .onnx.data
file) so the deployment goes live and the problem is visible in /health
and in the logs; /moderate returns 503 until the model loads.
"""

from __future__ import annotations

import logging
import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("content-moderation")

MODEL_PATH = "content_moderation_gpu.onnx"
DATA_PATH = "content_moderation_gpu.onnx.data"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
# Required size of the external weights file, computed from the ONNX graph:
# fc1.bias (1024) + fc2.weight (1024) + fc1.weight (30,720,000) + 65,536 offset.
EXPECTED_DATA_SIZE = 30_785_536

app = FastAPI(title="Content Moderation Model", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://web.pixelabs.in",
        "https://pixelabs.in",
        "https://www.pixelabs.in",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorizer = None
vectorizer_error: str | None = None
session = None
model_error: str | None = None


class ModerateRequest(BaseModel):
    text: str = Field(..., min_length=0, max_length=20000)


def load_vectorizer() -> None:
    global vectorizer, vectorizer_error
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        n = len(getattr(vectorizer, "vocabulary_", {}))
        log.info("Loaded %s (vocabulary: %d features)", VECTORIZER_PATH, n)
    except Exception as exc:  # noqa: BLE001
        vectorizer_error = f"{type(exc).__name__}: {exc}"
        log.exception("Failed to load vectorizer")


def load_model() -> None:
    global session, model_error
    try:
        actual = os.path.getsize(DATA_PATH) if os.path.exists(DATA_PATH) else 0
        if actual < EXPECTED_DATA_SIZE:
            raise RuntimeError(
                f"{DATA_PATH} is truncated: {actual} bytes present, "
                f"{EXPECTED_DATA_SIZE} bytes required (the original upload "
                "stopped at exactly 4 MiB). Re-upload the full file."
            )
        import onnxruntime as ort

        session = ort.InferenceSession(
            MODEL_PATH, providers=["CPUExecutionProvider"]
        )
        log.info("Loaded %s", MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        model_error = f"{type(exc).__name__}: {exc}"
        log.error("Model not available: %s", model_error)


load_vectorizer()
load_model()


@app.get("/")
def health() -> dict:
    return {
        "service": "content-moderation-model",
        "status": "ok" if (vectorizer is not None and session is not None) else "degraded",
        "vectorizer_loaded": vectorizer is not None,
        "model_loaded": session is not None,
        "vectorizer_error": vectorizer_error,
        "model_error": model_error,
        "expected_data_bytes": EXPECTED_DATA_SIZE,
    }


@app.post("/moderate")
def moderate(req: ModerateRequest) -> dict:
    if vectorizer is None:
        raise HTTPException(status_code=503, detail=f"Vectorizer unavailable: {vectorizer_error}")
    if session is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {model_error}")

    features = vectorizer.transform([req.text]).toarray().astype(np.float32)
    (logit,) = session.run(["output"], {"input": features})
    score = float(1.0 / (1.0 + np.exp(-float(logit[0][0]))))  # sigmoid
    return {
        "text_length": len(req.text),
        "toxic": score >= 0.5,
        "score": round(score, 6),
        "label": "toxic" if score >= 0.5 else "neutral",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
