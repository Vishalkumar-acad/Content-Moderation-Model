"""
Content Moderation Model — FastAPI inference server (for Render.com).

Loads the TF-IDF vectorizer (tfidf_vectorizer.pkl) and the ONNX model
(content_moderation_gpu.onnx, weights in content_moderation_gpu.onnx.data)
and exposes:

  GET  /         -> service health + model status
  POST /moderate -> {"text": "..."} -> toxicity score

The 29.4 MB weights file is shipped as a GitHub release asset (the copy
committed to git was truncated at 4 MiB). On startup, if the local copy is
missing or too small, it is downloaded automatically in a background
thread — the web server binds its port immediately, so Render's health
check passes even while the download is in flight. /moderate returns 503
until the model finishes loading.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import urllib.request

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
EXPECTED_SHA256 = "67a0c7e79e86d6bfc488b6dcb83b16a02fd929f91afe4c9fdd837d75d829a489"
RELEASE_URL = (
    "https://github.com/Vishalkumar-acad/Content-Moderation-Model/"
    "releases/download/model-v1/content_moderation_gpu.onnx.data"
)

app = FastAPI(title="Content Moderation Model", version="1.3.0")

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
model_status = "initializing"  # initializing | downloading | ok | error

# Deterministic profanity layer: the TF-IDF model is weak on very short texts
# (e.g. a single abusive word scores 0.001), so explicitly abusive words are
# always treated as toxic regardless of the model score.
_PROFANITY_RE = re.compile(
    r"\b(?:"
    r"asshole|assholes|bastard|bastards|bitch|bitches|bullshit|crap|cunt|cunts|"
    r"dick|dickhead|dickheads|dumbass|dumbasses|fag|faggot|faggots|fuck|fucked|"
    r"fucker|fuckers|fucking|motherfucker|motherfuckers|prick|pricks|puny cock|"
    r"retard|retarded|shit|shits|shithead|shitheads|shitty|slut|sluts|"
    r"twat|wanker|wankers|whore|whores|chutiya|chutiye|madarchod|behenchod|"
    r"bhosdike|bhen ke lode|gaandu|gandu|harami|kutta|kutte|haramkhor|"
    r"randi|rand|saala harami"
    r")\b",
    re.IGNORECASE,
)

# Deterministic harassment-phrase layer: politely-worded harassment
# ("nobody cares about your existence") contains no profanity, so the TF-IDF
# model scores it ~0.0005 — deep in "clean" territory, far below any usable
# threshold. Common harassment phrasings are therefore matched explicitly too.
_HARASSMENT_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:nobody|no\s?one|none)\s+(?:cares?|liked|likes|loves?|misses|wants|needs|asked)\s+(?:about\s+|for\s+)?(?:you|u|your|ur)\b",
        r"\b(?:nobody|no\s?one)\s+would\s+(?:even\s+)?(?:miss|notice|remember|care)\s+(?:you|u)\b",
        r"\b(?:world|planet|everyone|everybody)\s+would\s+be\s+better\s+(?:off\s+)?without\s+(?:you|u)\b",
        r"\b(?:everyone|everybody|all)\s+(?:hates|hate)\s+(?:you|u)\b",
        r"\bwaste\s+of\s+(?:oxygen|air|space)\b",
        r"\bkill\s+(?:yourself|urself)\b|\bkys\b|\bunalive\s+(?:yourself|urself)\b|\bend\s+(?:your|ur)\s+life\b",
        r"\bgo\s+die\b|\bdrop\s+dead\b|\b(?:you|u)\s+(?:should|deserve)\s+(?:to\s+)?die\b|\bdeserve\s+to\s+die\b",
        r"\b(?:you\s+are|you'?re|u\s+are)\s+(?:so\s+|such\s+)?(?:a\s+|an\s+)?(?:worthless|pathetic|useless|disgusting|repulsive|mistake|trash|garbage|shameful)\b",
        r"\byou\s+should\s+be\s+ashamed\b",
        r"\bwhy\s+don'?t\s+you\s+(?:just\s+)?(?:stop|quit|leave|disappear|go\s+away)\b",
        r"\bdo\s+(?:us|me)\s+a\s+favor\b[^.!?]{0,60}?\b(?:stop|quit|leave|disappear|die|delete)\b",
        r"\bnobody\s+asked\s+(?:you|u)\b",
        r"\bshouldn'?t\s+(?:exist|be\s+alive|be\s+born)\b",
        r"\bdon'?t\s+deserve\s+(?:to\s+live|life|anything|friends|love)\b",
        r"\b(?:your|ur)\s+(?:writing|posts?|content|art|work|existence|opinions?)\s+(?:is|are)\s+(?:garbage|trash|worthless|pathetic|pointless|useless)\b",
    )
]


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


def _local_data_ok() -> bool:
    try:
        return os.path.getsize(DATA_PATH) >= EXPECTED_DATA_SIZE
    except OSError:
        return False


def _try_load_model() -> bool:
    """Load the ONNX session from the local files. Returns True on success."""
    global session, model_error
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        model_error = None
        log.info("Loaded %s", MODEL_PATH)
        return True
    except Exception as exc:  # noqa: BLE001
        session = None
        model_error = f"{type(exc).__name__}: {exc}"
        log.error("Model not available: %s", model_error)
        return False


def _download_and_load() -> None:
    global model_status, model_error
    model_status = "downloading"
    try:
        log.info("Downloading model weights from release asset (%d bytes)...", EXPECTED_DATA_SIZE)
        tmp_path = DATA_PATH + ".tmp"
        urllib.request.urlretrieve(RELEASE_URL, tmp_path)
        size = os.path.getsize(tmp_path)
        if size != EXPECTED_DATA_SIZE:
            raise RuntimeError(f"downloaded file is {size} bytes, expected {EXPECTED_DATA_SIZE}")
        with open(tmp_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != EXPECTED_SHA256:
            raise RuntimeError(f"sha256 mismatch: got {digest}")
        os.replace(tmp_path, DATA_PATH)
        log.info("Weights downloaded and verified (sha256 ok).")
        if _try_load_model():
            model_status = "ok"
        else:
            model_status = "error"
    except Exception as exc:  # noqa: BLE001
        model_error = f"{type(exc).__name__}: {exc}"
        model_status = "error"
        log.error("Weight download failed: %s", model_error)


def load_model() -> None:
    global model_status
    if _local_data_ok() and _try_load_model():
        model_status = "ok"
        return
    if not _local_data_ok():
        log.warning(
            "%s missing or truncated (git copy was cut at 4 MiB); "
            "fetching full file from the GitHub release in the background.",
            DATA_PATH,
        )
        threading.Thread(target=_download_and_load, daemon=True).start()


load_vectorizer()
load_model()


@app.get("/")
def health() -> dict:
    return {
        "service": "content-moderation-model",
        "status": "ok" if (vectorizer is not None and session is not None) else "degraded",
        "vectorizer_loaded": vectorizer is not None,
        "model_loaded": session is not None,
        "model_status": model_status,
        "vectorizer_error": vectorizer_error,
        "model_error": model_error,
        "expected_data_bytes": EXPECTED_DATA_SIZE,
    }


@app.post("/moderate")
def moderate(req: ModerateRequest) -> dict:
    if vectorizer is None:
        raise HTTPException(status_code=503, detail=f"Vectorizer unavailable: {vectorizer_error}")
    if _PROFANITY_RE.search(req.text):
        return {
            "text_length": len(req.text),
            "toxic": True,
            "score": 1.0,
            "label": "toxic",
            "matched_rule": "profanity_list",
        }
    if any(r.search(req.text) for r in _HARASSMENT_RES):
        return {
            "text_length": len(req.text),
            "toxic": True,
            "score": 1.0,
            "label": "toxic",
            "matched_rule": "harassment_pattern",
        }
    if session is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable ({model_status}): {model_error}",
        )

    features = vectorizer.transform([req.text]).toarray().astype(np.float32)
    (prob,) = session.run(["output"], {"input": features})
    score = float(prob[0][0])  # graph already ends with Sigmoid — this is the probability
    return {
        "text_length": len(req.text),
        "toxic": score >= 0.5,
        "score": round(score, 6),
        "label": "toxic" if score >= 0.5 else "neutral",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
