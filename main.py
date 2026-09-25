"""
Content Moderation Model — FastAPI inference server (for Render.com).

v3 pipeline (DistilBERT deep learning):
  1. profanity wordlist (deterministic, v1.4.1) — with de-obfuscation
     normalization (leetspeak, fullwidth/homoglyphs, elongation, spaced
     single letters, zero-width chars)
  2. harassment-phrase regex (deterministic) — same normalization
  3. DistilBERT int8 ONNX model (tokenizers + onnxruntime)

The model (67 MB), tokenizer.json and tokenizer_config.json are shipped as
GitHub release assets (model-v3). On startup every file is verified against
its exact size and sha256; anything missing, truncated or stale (e.g. an old
v2 copy) is re-downloaded automatically — the 67 MB model in a background
thread, so the web server binds its port immediately and Render's health
check passes even while the download is in flight. /moderate returns 503
until the model finishes loading.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import unicodedata
import urllib.request

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("content-moderation")

MODEL_PATH = "distilbert_moderation_v3_int8.onnx"
MODEL_SIZE = 67_363_334
MODEL_SHA256 = "bf5981da82aad9aa0c7c4dbcf158480772d6111df28e42f69c99f72a27e76f3e"
TOKENIZER_PATH = "tokenizer.json"
TOKENIZER_SIZE = 711_661
TOKENIZER_SHA256 = "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"
TOKENIZER_CONFIG_PATH = "tokenizer_config.json"
TOKENIZER_CONFIG_SIZE = 322
TOKENIZER_CONFIG_SHA256 = "797ed9ba72b500001971b827b0040b8743def8ec38f9cd4cda4c4945734d3596"
RELEASE_BASE_URL = (
    "https://github.com/Vishalkumar-acad/Content-Moderation-Model/"
    "releases/download/model-v3/"
)
MAX_LEN = 128

app = FastAPI(title="Content Moderation Model", version="1.6.0")

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

tokenizer = None
session = None
model_error: str | None = None
model_status = "initializing"  # initializing | downloading | ok | error

# Deterministic profanity layer: short single-word abuse ("NIGGER") can still
# land under the model threshold on rare tokenizations, so explicitly abusive
# words are always treated as toxic regardless of the model score.
_PROFANITY_RE = re.compile(
    r"\b(?:"
    r"asshole|assholes|bastard|bastards|bitch|bitches|bullshit|crap|cunt|cunts|"
    r"dick|dickhead|dickheads|dumbass|dumbasses|assclown|assclowns|fag|faggot|"
    r"faggots|fuck|fucked|fucker|fuckers|fucking|motherfucker|motherfuckers|"
    r"prick|pricks|puny cock|retard|retarded|shit|shits|shithead|shitheads|"
    r"shitty|horseshit|slut|sluts|twat|scumbag|scumbags|nigger|niggas|nigga|"
    r"wanker|wankers|whore|whores|chutiya|chutiye|madarchod|behenchod|"
    r"bhosdike|bhen ke lode|gaandu|gandu|harami|kutta|kutte|haramkhor|"
    r"randi|rand|saala harami"
    r")\b",
    re.IGNORECASE,
)

# Deterministic harassment-phrase layer: politely-worded harassment
# ("nobody cares about your existence") contains no profanity, so a
# text classifier can still score it low. Common harassment phrasings are
# therefore matched explicitly too.
_HARASSMENT_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:nobody|no\s?one|none)\s+(?:cares?|liked|likes|loves?|misses|wants|needs|asked)\s+(?:about\s+|for\s+)?(?:you|u|ur)\b",
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

# --- De-obfuscation normalization (deterministic layers only) ---------------
# The wordlist and harassment regexes are matched against BOTH the original
# text and a normalized copy, so common evasion tricks still get caught:
#   "sh1t", "f u c k", "ｆｕｃｋ" (fullwidth), "fuuuuck", zero-width padding.
# The transformer always receives the ORIGINAL text (it was trained on real
# text; normalized input would shift its scores unpredictably).
_LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
    "@": "a", "$": "s", "!": "i", "+": "t",
})
_ZERO_WIDTH_RE = re.compile("[\u200b-\u200f\u2060-\u2064\ufeff\u00ad]")
# a run of 3+ whitespace-separated 1-2 char tokens: "f u c k", "sh i i t" -> "fuck", "shiit"
_SPACED_RUN_RE = re.compile(r"\b\w{1,2}\b(?:\s+\b\w{1,2}\b)+")
# collapse 3+ repeats of a char ("fuuuck"->"fuck"); 2-char doubles like the
# "ss" in "asshole" or "ll" in "kill" must be preserved in the standard
# variant or the patterns themselves would stop matching
_ELONGATION_RE = re.compile(r"(.)\1{2,}")
# the aggressive variant additionally collapses 2-char doubles
# ("shiit"->"shit"); double-letter words like "asshole" are still caught on
# the original/standard variants, so this only adds coverage
_ELONGATION_RE_AGGRESSIVE = re.compile(r"(.)\1+")


def _deobfuscate(text: str) -> str:
    """Normalize common obfuscation tricks for deterministic-layer matching."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _ZERO_WIDTH_RE.sub("", t)
    t = t.translate(_LEET_MAP)
    t = _SPACED_RUN_RE.sub(lambda m: re.sub(r"\s+", "", m.group(0)), t)
    t = _ELONGATION_RE.sub(r"\1", t)
    return t


def _deobfuscate_aggressive(norm: str) -> str:
    """Extra pass for spacing+elongation combos ("sh i i t" -> "shit")."""
    return _ELONGATION_RE_AGGRESSIVE.sub(r"\1", norm)


class ModerateRequest(BaseModel):
    text: str = Field(..., min_length=0, max_length=20000)


def _file_ok(path: str, size: int, sha256: str) -> bool:
    """True only if the local file exists with the exact size and sha256."""
    try:
        if os.path.getsize(path) != size:
            return False
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest() == sha256
    except OSError:
        return False


def _download_verified(path: str, size: int, sha256: str) -> None:
    """Download `path` from the model-v3 release, verify size+sha, move into place."""
    url = RELEASE_BASE_URL + path
    tmp_path = path + ".tmp"
    urllib.request.urlretrieve(url, tmp_path)
    got = os.path.getsize(tmp_path)
    if got != size:
        raise RuntimeError(f"downloaded {path} is {got} bytes, expected {size}")
    with open(tmp_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"sha256 mismatch for {path}: got {digest}")
    os.replace(tmp_path, path)


def _try_load_all() -> bool:
    """Load tokenizer + ONNX session from the local files. Returns True on success."""
    global tokenizer, session, model_error
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(TOKENIZER_PATH)
        # Explicit padding/truncation — NEVER handcraft the attention mask;
        # the tokenizer's own encoding is the source of truth (v3 lesson).
        tok.enable_truncation(max_length=MAX_LEN)
        tok.enable_padding(length=MAX_LEN, pad_id=0, pad_token="[PAD]")
        tokenizer = tok

        import onnxruntime as ort

        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        # warmup so the first real request is fast
        _predict("warmup: please take out the garbage")
        model_error = None
        log.info("Loaded tokenizer + %s (warmup ok)", MODEL_PATH)
        return True
    except Exception as exc:  # noqa: BLE001
        tokenizer = None
        session = None
        model_error = f"{type(exc).__name__}: {exc}"
        log.error("Model not available: %s", model_error)
        return False


def _ensure_and_load() -> None:
    global model_status, model_error
    model_status = "downloading"
    try:
        files = [
            (TOKENIZER_CONFIG_PATH, TOKENIZER_CONFIG_SIZE, TOKENIZER_CONFIG_SHA256),
            (TOKENIZER_PATH, TOKENIZER_SIZE, TOKENIZER_SHA256),
            (MODEL_PATH, MODEL_SIZE, MODEL_SHA256),
        ]
        for path, size, sha in files:
            if not _file_ok(path, size, sha):
                log.info("Fetching %s from model-v3 release (%d bytes)...", path, size)
                _download_verified(path, size, sha)
                log.info("%s downloaded and verified (sha256 ok).", path)
        if _try_load_all():
            model_status = "ok"
        else:
            model_status = "error"
    except Exception as exc:  # noqa: BLE001
        model_error = f"{type(exc).__name__}: {exc}"
        model_status = "error"
        log.error("Model download failed: %s", model_error)


def load_model() -> None:
    global model_status
    files_ok = all([
        _file_ok(TOKENIZER_CONFIG_PATH, TOKENIZER_CONFIG_SIZE, TOKENIZER_CONFIG_SHA256),
        _file_ok(TOKENIZER_PATH, TOKENIZER_SIZE, TOKENIZER_SHA256),
        _file_ok(MODEL_PATH, MODEL_SIZE, MODEL_SHA256),
    ])
    if files_ok and _try_load_all():
        model_status = "ok"
        return
    log.warning(
        "v3 model files missing or stale; fetching from the GitHub release "
        "in the background (67 MB download)."
    )
    threading.Thread(target=_ensure_and_load, daemon=True).start()


def _predict(text: str) -> float:
    """DistilBERT int8 score for one text (0..1, sigmoid included in graph)."""
    enc = tokenizer.encode(text)
    ids = np.array([enc.ids], dtype=np.int64)
    masks = np.array([enc.attention_mask], dtype=np.int64)
    (score,) = session.run(["score"], {"input_ids": ids, "attention_mask": masks})
    return float(score[0][0])


load_model()


@app.get("/")
def health() -> dict:
    return {
        "service": "content-moderation-model",
        "version": "1.6.0 (DistilBERT v3)",
        "status": "ok" if (tokenizer is not None and session is not None) else "degraded",
        "tokenizer_loaded": tokenizer is not None,
        "model_loaded": session is not None,
        "model_status": model_status,
        "model_error": model_error,
        "expected_model_bytes": MODEL_SIZE,
    }


@app.post("/moderate")
def moderate(req: ModerateRequest) -> dict:
    # Deterministic layers run on the original text plus two de-obfuscated
    # variants, so "sh1t" / "f u c k" / "sh i i t" / "ｆｕｃｋ" are caught too.
    norm = _deobfuscate(req.text)
    for variant in (req.text, norm, _deobfuscate_aggressive(norm)):
        if _PROFANITY_RE.search(variant):
            return {
                "text_length": len(req.text),
                "toxic": True,
                "score": 1.0,
                "label": "toxic",
                "matched_rule": "profanity_list",
            }
        if any(r.search(variant) for r in _HARASSMENT_RES):
            return {
                "text_length": len(req.text),
                "toxic": True,
                "score": 1.0,
                "label": "toxic",
                "matched_rule": "harassment_pattern",
            }
    if session is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable ({model_status}): {model_error}",
        )

    score = _predict(req.text)
    return {
        "text_length": len(req.text),
        "toxic": score >= 0.5,
        "score": round(score, 6),
        "label": "toxic" if score >= 0.5 else "neutral",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
