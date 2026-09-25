# Content-Moderation-Model

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)

A lightweight, fast content-moderation model for real-time comment and post moderation. A TF-IDF + MLP classifier trained on 1.75M public comment conversations from the Jigsaw/Civil Comments toxicity dataset, exported to ONNX for fast CPU inference and served through a small FastAPI service.

**Current version: v2** — an 80/20 anti-forgetting retrain of the original model, trained on error-mined additions plus the full original dataset.

## How it works — three layers

1. **Profanity wordlist (deterministic):** common English and Hindi abusive words are always blocked instantly (score 1.0, `matched_rule: profanity_list`), regardless of the model's output. This covers very short texts where statistical models are weak.
2. **Harassment-phrase patterns (deterministic):** politely-worded harassment ("nobody likes you", "you are worthless", "kill yourself", ...) is matched explicitly (score 1.0, `matched_rule: harassment_pattern`).
3. **TF-IDF + MLP classifier (96.0% accuracy):** the ONNX model scores the text; `score >= 0.5` is labeled toxic. Typical scores are strongly bimodal — clean text ≈ 0.001–0.02, clearly abusive text ≈ 0.99–1.0.

## Quick start (local)

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

On first start, the model weights and the TF-IDF vectorizer are downloaded
automatically from the
[model-v2 release](https://github.com/Vishalkumar-acad/Content-Moderation-Model/releases/tag/model-v2)
and each file is verified (exact size + sha256); a stale v1 copy is rejected
and re-downloaded. The API docs are served at `/docs`.

## API

### `GET /` — health / status

```json
{"service": "content-moderation-model", "status": "ok", "model_loaded": true, "model_status": "ok"}
```

### `POST /moderate`

```bash
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "you are a wonderful person"}'
```

```json
{"text_length": 26, "toxic": false, "score": 0.0185, "label": "neutral"}
```

## Deployment (Render.com)

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

`render.yaml` is included. The service downloads its own weights at startup, so
no large files need to be in the git tree. On Render's free plan the instance
sleeps when idle; the first request after idle takes ~50 s to wake up.

## Training

Trained in a Kaggle GPU notebook (PyTorch), then exported to ONNX.

### v2 (current) — 80/20 anti-forgetting retrain

| | |
|---|---|
| Dataset | [SetFit/toxic_conversations](https://huggingface.co/datasets/SetFit/toxic_conversations) — 1,754,874 comments, 100% retained **plus** ~90 unique error rows (mined false positives/negatives + curated anchors), oversampled to ~25% of the mix |
| Method | 80/20 rule: all original data kept, error additions oversampled — the model learns the new failure families without forgetting the old ones (best-val checkpoint guards against overfitting) |
| Vectorizer | `TfidfVectorizer(max_features=30_000, ngram_range=(1, 2))`, scikit-learn 1.6.1 (refit on the combined corpus) |
| Architecture | `fc1` Linear 30000→256 → ReLU → Dropout(0.3) → `fc2` Linear 256→1 → Sigmoid (identical to v1) |
| Objective | `BCELoss`, Adam (lr=0.002), 12 epochs, batch 4096, best-val checkpointing |
| Best val AUC | **0.9825** |
| Held-out accuracy | **95.96%** (v1: 94.4%) |
| Export | `torch.onnx.export` (external weights) + `joblib.dump(vectorizer)` |

### v1 (original)

| | |
|---|---|
| Dataset | [SetFit/toxic_conversations](https://huggingface.co/datasets/SetFit/toxic_conversations) — 1.8M comments (~8% toxic) |
| Objective | `BCELoss`, Adam (lr=0.005), 3 epochs |
| Held-out accuracy | 94.4% |

## What v2 fixed

The v2 retrain was driven by errors found by an automated error-mining run
(20,000 real civil-comments scored through the live pipeline) and real
user reports:

| Case | v1 score | v2 score |
|---|---|---|
| "Please put the garbage in the trash" (benign) | 0.999999 | 0.000253 |
| "He works as a garbage collector in our city" (benign) | 0.963 | 0.000068 |
| "My phone battery is trash" (benign) | 0.87 | 0.000017 |
| "You are totally trash" (toxic) | 0.944 | 0.999992 |
| "You are a worthless piece of garbage" (toxic) | ~0.9 | 1.0000 |

The v1.4.1 wordlist update additionally catches (score 1.0):
nigger/niggas/nigga, horseshit (word-boundary fix), assclown, scumbag.

### Independent evaluation (20,000 fresh civil comments, never seen in training)

| | v1 | v2 model-only | v2 full pipeline |
|---|---|---|---|
| Accuracy | ~95.7% | **99.11%** | **98.89%** |
| AUC | ~0.978 | **0.9858** | 0.9866 |
| Missed toxic (FN) | 842 (4.21%) | **99 (0.50%)** | **97 (0.48%)** |
| False positives | 13 | 80 (0.40%) | 124 (0.62%)* |

\* The full pipeline also flags comments that contain profanity but were
labeled non-toxic by the dataset (profanity != toxicity) — a deliberate
policy choice. Most full-pipeline false positives land in the 0.5-0.9
flag-only zone (admin review), not auto-hide (>= 0.9).

## Model details

| | |
|---|---|
| Architecture | TF-IDF (30,000 features) → Gemm fc1 (30000→256) → ReLU → Gemm fc2 (256→1) → Sigmoid |
| Size | 1 KB graph + 30,785,536 bytes external weights (~29 MB) |
| Inference | ONNX Runtime (CPU), milliseconds per request |

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
