"""
Content Moderation Model — FastAPI inference server (for Render.com).

v4.1 pipeline (DistilBERT + context-aware deterministic layers):
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

MODEL_PATH = "distilbert_moderation_v4_int8.onnx"
MODEL_SIZE = 67_362_849
MODEL_SHA256 = "715ad02d9ea23de95360e5f0b18377a5a029414fa81c4e8ff7d30365abd7d69a"
TOKENIZER_PATH = "tokenizer.json"
TOKENIZER_SIZE = 711_661
TOKENIZER_SHA256 = "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"
TOKENIZER_CONFIG_PATH = "tokenizer_config.json"
TOKENIZER_CONFIG_SIZE = 351
TOKENIZER_CONFIG_SHA256 = "e1c2a61a99bda00f6c55303a210b30e2f92dcf8b555e215812e2eb583e177ffd"
RELEASE_BASE_URL = (
    "https://github.com/Vishalkumar-acad/Content-Moderation-Model/"
    "releases/download/model-v4/"
)
MAX_LEN = 128

app = FastAPI(title="Content Moderation Model", version="2.1.0")

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
# =====================================================================
# v4.1 deterministic layers -- CONTEXT-AWARE
# =====================================================================
# Person-directed slurs / insults: abusive only when actually aimed at a
# person; quoted / reported / discussed / negated mentions are not.
_SLUR_RE = re.compile(
    r"\b(?:asshole|assholes|bastard|bastards|bitch|bitches|cunt|cunts|dick|dickhead|"
    r"dickheads|dumbass|dumbasses|assclown|assclowns|fag|faggot|faggots|motherfucker|"
    r"motherfuckers|prick|pricks|retard|retarded|slut|sluts|twat|scumbag|scumbags|"
    r"nigger|niggas|nigga|wanker|wankers|whore|whores|chutiya|chutiye|madarchod|"
    r"behenchod|bhosdike|bhen ke lode|gaandu|gandu|harami|haramkhor|randi|kutta|kutte)\b",
    re.IGNORECASE,
)
# Expletives / intensifiers: abusive only when directed at a person
# ("fuck you"), not as a filler ("this fucking traffic").
_EXPLETIVE_RE = re.compile(
    r"\b(?:fuck|fucked|fucker|fuckers|fucking|shit|shits|shitty|bullshit|horseshit|crap)\b",
    re.IGNORECASE,
)
# Constructions that are always directed abuse.
_DIRECTED_RE = re.compile(
    r"\bfuck\s+(?:you|u|off)\b|\bshut\s+(?:the\s+fuck\s+)?up\b|\bstfu\b",
    re.IGNORECASE,
)
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
# Threats: strong = clear harm (toxic); weak = ambiguous (review zone).
_THREAT_STRONG_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi\s+know\s+where\s+you\s+(?:live|work|sleep|are|stay)\b",
        r"\b(?:i|we)\s+(?:will|'?ll|am\s+going\s+to|'?m\s+gonna|gonna)\s+(?:find|hunt|hurt|kill|come\s+after|get)\s+(?:you|u)\b",
        r"\b(?:your|ur)\s+(?:family|kids?|children|parents)\s+will\s+pay\b",
        r"\bsleep\s+with\s+one\s+eye\s+open\b",
        r"\bi'?m\s+coming\s+for\s+you\b",
        r"\b(?:i'?ll|i\s+will)\s+make\s+you\s+pay\b",
        r"\byou\s+(?:'?re|are)\s+(?:dead|done\s+for)\b",
        r"\bwatch\s+your\s+back\b",
    )
]
_THREAT_WEAK_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\byou\s+(?:will|'?ll|are\s+going\s+to|gonna)\s+regret\s+(?:this|it)\b",
        r"\byou'?ll\s+(?:be\s+sorry|find\s+out)\b",
        r"\bfind\s+out\s+what\s+happens\b",
        r"\byou\s+better\s+(?:watch|be\s+careful|sleep)\b",
        r"\benjoy\s+your\s+last\s+(?:days|hours|moments)\b",
        r"\byour\s+days\s+are\s+numbered\b",
    )
]
_QUOTE_RE = re.compile(r'"[^"]*"|\u201c[^\u201d]*\u201d|\'[^\']*\'')
_NEG_RE = re.compile(r"\b(?:not|never|no|n't|without|nor|neither)\b", re.IGNORECASE)
_REPORT_RE = re.compile(
    r"\b(?:the\s+word|the\s+term|called|calling|says?|said|saying|wrote|writing|quoting|"
    r"quote[ds]?|report(?:ed|s|ing)?|discuss(?:ed|ing)?|example\s+of|mention(?:ed|ing|s)?|"
    r"slurs?|evidence|according\s+to)\b",
    re.IGNORECASE,
)
_YOU_RE = re.compile(r"\b(?:you|u|your|ur|you'?re|yourself|urself|yours)\b", re.IGNORECASE)
_INSULT_LEX = re.compile(
    r"\b(?:stupid|idiots?|idiotic|dumbs?|fools?|foolish|morons?|moronic|imbecile|trash|garbage|"
    r"worthless|pathetic|useless|disgusting|repulsive|losers?|scum|jerks?|hate)\b",
    re.IGNORECASE,
)
# Benign uses of "kill" (not aimed at a person).
_BENIGN_KILL_RE = re.compile(
    r"\bcould\s+kill\s+for\b|\bkill\s+(?:time|the\s+(?:process|background|engine|lights|"
    r"project|bug|server|connection|power|meeting))\b|\bkilled\s+(?:it|the\s+(?:bacteria|virus|"
    r"plants|engine|project|bug|lights|server))\b|\bkilling\s+(?:me|it)\b",
    re.IGNORECASE,
)
REVIEW_SCORE = 0.6
NEUTRAL_CAP = 0.3


def _quoted_spans(text):
    return [(m.start(), m.end()) for m in _QUOTE_RE.finditer(text)]


def _in_quotes(spans, start, end):
    return any(a <= start and end <= b for a, b in spans)


def _context(text, start, end, spans):
    """'clear' = quoted+reported / meta / negated (exempt). 'ambiguous' = quoted
    only (review). None = no exemption."""
    if _NEG_RE.search(text[max(0, start - 40):start]):
        return "clear"
    window = text[max(0, start - 80): min(len(text), end + 40)]
    if _in_quotes(spans, start, end):
        return "clear" if _REPORT_RE.search(window) else "ambiguous"
    if _REPORT_RE.search(window):
        return "clear"
    return None


def _directed(text, start, end):
    return bool(_YOU_RE.search(text[max(0, start - 25): min(len(text), end + 25)]))


def _model_context(text, spans):
    found = clear = amb = False
    for rx in (_INSULT_LEX, _SLUR_RE):
        for m in rx.finditer(text):
            found = True
            ctx = _context(text, m.start(), m.end(), spans)
            if ctx == "clear":
                clear = True
            elif ctx == "ambiguous":
                amb = True
            else:
                return None          # unguarded insult/slur -> let the model decide
    for m in _EXPLETIVE_RE.finditer(text):
        found = True
        ctx = _context(text, m.start(), m.end(), spans)
        if ctx == "clear":
            clear = True
        elif _directed(text, m.start(), m.end()):
            return None              # directed expletive -> let the model decide
        else:
            clear = True             # untargeted frustration -> neutral signal
    if _BENIGN_KILL_RE.search(text):
        found = True
        clear = True
    if not found:
        return None
    return "clear" if clear else ("ambiguous" if amb else None)


def _hit(text, rule, score=1.0):
    return {"text_length": len(text), "toxic": True, "score": score,
            "label": "toxic", "matched_rule": rule}


def _review(text, rule):
    return {"text_length": len(text), "toxic": True, "score": REVIEW_SCORE,
            "label": "toxic", "matched_rule": rule, "review": True}


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
        "v4 model files missing or stale; fetching from the GitHub release "
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
        "version": "2.1.0 (DistilBERT v4.1 + context layers)",
        "status": "ok" if (tokenizer is not None and session is not None) else "degraded",
        "tokenizer_loaded": tokenizer is not None,
        "model_loaded": session is not None,
        "model_status": model_status,
        "model_error": model_error,
        "expected_model_bytes": MODEL_SIZE,
    }


@app.post("/moderate")
def moderate(req: ModerateRequest) -> dict:
    text = req.text
    norm = _deobfuscate(text)
    aggressive = _deobfuscate_aggressive(norm)
    spans = _quoted_spans(text)
    review = False

    def ctx_of(m):
        return _context(text, m.start(), m.end(), spans)

    # 1) directed abuse + harassment (targeted at a person)
    for rx in [_DIRECTED_RE] + _HARASSMENT_RES:
        for m in rx.finditer(text):
            c = ctx_of(m)
            if c == "clear":
                continue
            if c == "ambiguous":
                review = True
                continue
            return _hit(text, "harassment_pattern")

    # 2) strong threats -> toxic
    for rx in _THREAT_STRONG_RES:
        for m in rx.finditer(text):
            c = ctx_of(m)
            if c == "clear":
                continue
            if c == "ambiguous":
                review = True
                continue
            return _hit(text, "threat_pattern", 0.9)

    # 3) weak threats -> review zone
    for rx in _THREAT_WEAK_RES:
        for m in rx.finditer(text):
            if ctx_of(m) == "clear":
                continue
            review = True

    # 4) slurs (person-directed)
    for m in _SLUR_RE.finditer(text):
        c = ctx_of(m)
        if c == "clear":
            continue
        if c == "ambiguous":
            review = True
            continue
        return _hit(text, "profanity_list")

    # 5) expletives -- only abusive when directed
    for m in _EXPLETIVE_RE.finditer(text):
        c = ctx_of(m)
        if c == "clear":
            continue
        if _directed(text, m.start(), m.end()):
            return _hit(text, "profanity_list")
        if c == "ambiguous":
            review = True

    # 6) obfuscation re-catch -- only when the original text was clean
    orig_has = bool(
        _DIRECTED_RE.search(text) or _SLUR_RE.search(text) or _EXPLETIVE_RE.search(text)
        or any(rx.search(text) for rx in _HARASSMENT_RES)
        or any(rx.search(text) for rx in _THREAT_STRONG_RES)
        or any(rx.search(text) for rx in _THREAT_WEAK_RES)
    )
    if not orig_has:
        for variant in (norm, aggressive):
            if any(rx.search(variant) for rx in _THREAT_STRONG_RES):
                return _hit(text, "threat_pattern", 0.9)
            if _DIRECTED_RE.search(variant):
                return _hit(text, "harassment_pattern")
            if _SLUR_RE.search(variant):
                return _hit(text, "profanity_list")
            for m in _EXPLETIVE_RE.finditer(variant):
                if _directed(variant, m.start(), m.end()):
                    return _hit(text, "profanity_list")

    # 7) model, with context guard
    if session is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable ({model_status}): {model_error}",
        )
    score = _predict(text)
    mctx = _model_context(text, spans)
    if mctx == "clear":
        score = min(score, NEUTRAL_CAP)
    elif mctx == "ambiguous":
        review = True
    if review:
        score = REVIEW_SCORE        # ambiguous -> review zone, never auto-hidden
    return {
        "text_length": len(text),
        "toxic": score >= 0.5,
        "score": round(score, 6),
        "label": "toxic" if score >= 0.5 else "neutral",
        "review": bool(review),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
