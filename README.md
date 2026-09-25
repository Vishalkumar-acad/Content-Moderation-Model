# Content-Moderation-Model

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)
![DistilBERT](https://img.shields.io/badge/Model-DistilBERT%20int8-FF6F00.svg)

A content-moderation model for real-time comment and post moderation, served through a small FastAPI service on CPU.

**Current version: v3** — a fine-tuned **DistilBERT** transformer (int8-quantized ONNX, 67 MB) that understands context, replacing the v2 TF-IDF classifier. Trained on 1.75M public comment conversations from the Jigsaw/Civil Comments toxicity dataset.

## How it works — three layers

1. **Profanity wordlist (deterministic):** common English and Hindi abusive words are always blocked instantly (score 1.0, `matched_rule: profanity_list`), regardless of the model's output. This covers very short texts where statistical models are weak.
2. **Harassment-phrase patterns (deterministic):** politely-worded harassment ("nobody likes you", "you are worthless", "kill yourself", ...) is matched explicitly (score 1.0, `matched_rule: harassment_pattern`).
3. **DistilBERT classifier (v3):** the ONNX model scores the text; `score >= 0.5` is labeled toxic. Typical scores are strongly bimodal — clean text ≈ 0.0003–0.005, clearly abusive text ≈ 0.92–1.0.

Layers 1–2 are also matched against a **de-obfuscated copy** of the text
(NFKC unicode normalization, a leetspeak digit map, collapsed character
elongation and spaced-out letters), so `sh1t`, `f u c k`, `sh i i t` or
fullwidth `ｆｕｃｋ` are caught like plain profanity. The transformer always
scores the original text — it was trained on real text and normalized input
would shift its scores unpredictably.

## Quick start (local)

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

On first start, the model, tokenizer and tokenizer config are downloaded
automatically from the
[model-v3 release](https://github.com/Vishalkumar-acad/Content-Moderation-Model/releases/tag/model-v3)
and each file is verified (exact size + sha256); a stale v2 copy is rejected
and re-downloaded. The API docs are served at `/docs`.

## API

### `GET /` — health / status

```json
{"service": "content-moderation-model", "version": "1.6.0 (DistilBERT v3)", "status": "ok", "model_loaded": true, "model_status": "ok"}
```

### `POST /moderate`

```bash
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "you are a wonderful person"}'
```

```json
{"text_length": 26, "toxic": false, "score": 0.0003, "label": "neutral"}
```

The raw `score` is always returned, so callers can apply their own zones
(e.g. `>= 0.9` auto-hide, `0.5–0.9` admin-review flag).

## Deployment (Render.com)

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

`render.yaml` is included. The service downloads its own model files at
startup (fail-closed sha256 verification), so no large files need to be in
the git tree. On Render's free plan the instance sleeps when idle; the first
request after idle takes ~50 s to wake up.

## v3 — DistilBERT (current)

| | |
|---|---|
| Dataset | [SetFit/toxic_conversations](https://huggingface.co/datasets/SetFit/toxic_conversations) — 1,754,874 comments (~8% toxic) |
| Base model | `distilbert-base-uncased` (HuggingFace transformers), max sequence 128 tokens |
| Training | Kaggle GPU (T4 x2), 3 epochs, best val AUC **0.9662** |
| Export | PyTorch → ONNX fp32 (268 MB) → dynamic int8 quantization (**67.4 MB**) |
| Inference | `input_ids` + `attention_mask` (int64, padded to 128) → `score` (sigmoid included), ONNX Runtime CPU |

### Why v3 — context understanding

The v1/v2 TF-IDF classifier could not tell *garbage the insult* from
*garbage the trash*. The transformer can:

| Case | v1 | v2 (TF-IDF) | v3 (DistilBERT) |
|---|---|---|---|
| "Please put the garbage in the trash" (benign) | 0.9999 | 0.0003 | **0.0024** |
| "He works as a garbage collector in our city" (benign) | 0.963 | 0.0001 | **0.0004** |
| "My phone battery is trash" (benign) | 0.87 | 0.0000 | **0.0005** |
| "You are totally trash" (toxic) | 0.944 | 1.0000 | **0.9955** |
| "You are a worthless piece of garbage" (toxic) | ~0.9 | 1.0000 | **1.0000** |

Remaining known borderline cases (documented, not hidden): "Please take
out the garbage" scores 0.52 and "I could kill for a cup of coffee right
now" scores 0.81 — both land in the admin-review zone, the comment stays
visible. No model is perfect; the sweep table below shows the measured
trade-off.

Known limitations (honest list):

- The wordlist layer is context-blind — an educational or quoted
  discussion of a slur still scores 1.0. The raw `score` field lets a
  caller distinguish model-judged toxicity from wordlist hits via
  `matched_rule`.
- Censored spellings ("f\*ck") and mixed-script homoglyphs (Cyrillic
  lookalikes) are not caught by the de-obfuscation layer.
- The API has no built-in authentication or rate limiting. CORS is
  restricted, but if you deploy it publicly, put it behind your own
  gateway.
- Multilingual coverage is English + some Hindi; romanized or other
  Indic-language abuse relies on the wordlist only.

### Independent evaluation — same 20,000 fresh civil comments

Both models scored through the **full production pipeline** (all three
layers) on identical, never-seen data:

| | v2 (TF-IDF) | v3 (DistilBERT) |
|---|---|---|
| Accuracy | **98.95%** | 96.84% |
| Toxic caught (TP) | 419/502 (83.5%) | **486/502 (96.8%)** |
| Missed toxic (FN) | 83 | **16 (5x better)** |
| Innocent flagged (FP) | **127 (0.65%)** | 616 (3.2%)* |
| AUC | 0.9894 | **0.9942** |

\* Most v3 false positives land in the 0.5–0.9 admin-review zone (comment
stays visible), not auto-hide — a deliberate recall-first policy. The
decision threshold was calibrated with a measured sweep:

| threshold | FP | FN | toxic caught |
|---|---|---|---|
| **0.5 (production)** | 617 | **16** | **96.8%** |
| 0.6 | 469 | 22 | 95.6% |
| 0.7 | 318 | 31 | 93.8% |
| 0.8 | 211 | 47 | 90.6% |
| 0.9 | 112 | 75 | 85.1% |

## v2 — TF-IDF retrain (previous)

An 80/20 anti-forgetting retrain of the original TF-IDF + MLP model: all
original data kept, ~90 error-mined additions (from a 20,000-comment mining
run + real user reports) oversampled to ~25% of the mix. Best val AUC
0.9825, held-out accuracy 95.96%. Weights remain available in the
[model-v2 release](https://github.com/Vishalkumar-acad/Content-Moderation-Model/releases/tag/model-v2)
as a rollback.

## v1 — original (historical)

TF-IDF (30,000 features) → Linear 30000→256 → ReLU → Linear 256→1 →
Sigmoid. Held-out accuracy 94.4%. The complete v1 snapshot (code +
vectorizer) lives at the
[`model-v1` tag](https://github.com/Vishalkumar-acad/Content-Moderation-Model/tree/model-v1),
and its full weights are published as the
[model-v1 release](https://github.com/Vishalkumar-acad/Content-Moderation-Model/releases/tag/model-v1)
asset (the copy once committed to the git tree was a truncated 4 MiB
upload, so the release asset is the real one).

## Files in this repo

| File | Purpose |
|---|---|
| `main.py` | Production FastAPI server (v1.6.0) — three-layer pipeline with de-obfuscation, fail-closed model download |
| `requirements.txt` | fastapi, uvicorn, onnxruntime, tokenizers, numpy (version-bounded) |
| `render.yaml` | Render.com service definition |
| `test-live-v3.py` | 24-case battery against the **live** production API |
| `colab-v2-vs-v3.py` | Head-to-head v2 vs v3 evaluation on the same 20k comments |
| `colab-v3-accuracy.py` | v3 full-pipeline accuracy on the same 20k data |
| `colab-v3-threshold-sweep.py` | FP/FN vs decision threshold (calibration data) |
| `kaggle-v3-train-distilbert.py` | v3 training notebook script (PyTorch) |
| `kaggle-v3-verify-deploy.py` | Post-export verification + int8 variants + deploy bundle |

## Release assets (model-v3)

| File | Size | sha256 |
|---|---|---|
| `distilbert_moderation_v3_int8.onnx` | 67,363,334 | `bf5981da82aad9aa0c7c4dbcf158480772d6111df28e42f69c99f72a27e76f3e` |
| `tokenizer.json` | 711,661 | `da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0` |
| `tokenizer_config.json` | 322 | `797ed9ba72b500001971b827b0040b8743def8ec38f9cd4cda4c4945734d3596` |

**Optional higher-precision variant (fp16, 134 MB)** — same model in
float16 instead of int8 (max score difference from the fp32 original:
0.0024). Useful if you want to re-quantize yourself or avoid int8
artifacts. Shipped as two parts because of the 100 MB asset limit:

| File | Size | sha256 |
|---|---|---|
| `distilbert_moderation_v3_fp16.onnx.part00` | 67,038,759 | `8ad12afa5a0adf1a7d636ec6d572a1c13eca683c854bd0095c1f1ea7fd550a38` |
| `distilbert_moderation_v3_fp16.onnx.part01` | 67,038,760 | `3d03bea5eba7fd5a3ac43b1bdba7eb7d506dc1e7d510d512da2ec429f95ecdd1` |

Join the parts and verify:

```bash
cat distilbert_moderation_v3_fp16.onnx.part00 distilbert_moderation_v3_fp16.onnx.part01 > distilbert_moderation_v3_fp16.onnx
# joined sha256: 5329703d8405fd90deccc4028b48c6b27f8122c1d553a36f30cb8fc03c591aba (134,077,519 bytes)
```

(Windows: `copy /b distilbert_moderation_v3_fp16.onnx.part00+distilbert_moderation_v3_fp16.onnx.part01 distilbert_moderation_v3_fp16.onnx`)

**Original fp32 training export (268 MB)** — the full-precision model
straight from training, for anyone who wants to re-quantize (e.g. with
different int8 settings) or continue fine-tuning:

| File | Size | sha256 |
|---|---|---|
| `distilbert_moderation_v3_fp32.onnx` | 267,940,094 | `674db19489e42d5c4a7ef64e6c56c07b15a3ab52306d1fe24077d5ebab95b4f0` |

Not needed to run the API — production serves the int8 build.

## Dataset & attribution

The training data is a version of the [Jigsaw Unintended Bias in Toxicity
Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
dataset (Civil Comments platform), via
[SetFit/toxic_conversations](https://huggingface.co/datasets/SetFit/toxic_conversations),
licensed **CC BY 4.0**. If you fork or reuse this model, please keep this
attribution intact.

## License

MIT — see [LICENSE](LICENSE). You are free to use, modify, and redistribute
this model and server; just keep the copyright notice intact.
