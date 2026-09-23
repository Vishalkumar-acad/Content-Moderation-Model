# Content-Moderation-Model

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)

A lightweight, fast content-moderation model for real-time comment and post moderation. A TF-IDF + MLP classifier trained on 1.8M public comment conversations from the Jigsaw/Civil Comments toxicity dataset, exported to ONNX for fast CPU inference and served through a small FastAPI service.

## How it works — two layers

1. **Profanity wordlist (deterministic):** common English and Hindi abusive words are always blocked instantly (score 1.0, `matched_rule: profanity_list`), regardless of the model's output. This covers very short texts where statistical models are weak.
2. **TF-IDF + MLP classifier (94.4% accuracy):** the ONNX model scores the text; `score >= 0.5` is labeled toxic. Typical scores are strongly bimodal — clean text ≈ 0.001–0.02, clearly abusive text ≈ 0.99–1.0.

## Quick start (local)

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

On first start, the ~29 MB model weights are downloaded automatically from the
[model-v1 release](https://github.com/Vishalkumar-acad/Content-Moderation-Model/releases/tag/model-v1)
in a background thread and verified (size + sha256). The API docs are served at `/docs`.

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

Trained in a Kaggle GPU notebook (PyTorch), then exported to ONNX:

| | |
|---|---|
| Dataset | [SetFit/toxic_conversations](https://huggingface.co/datasets/SetFit/toxic_conversations) — 1,804,874 comments (~8% toxic) |
| Vectorizer | `TfidfVectorizer(max_features=30_000, ngram_range=(1, 2))`, scikit-learn 1.6.1 |
| Architecture | `fc1` Linear 30000→256 → ReLU → Dropout(0.3) → `fc2` Linear 256→1 → Sigmoid |
| Objective | `BCELoss`, Adam optimizer (lr=0.005), 3 epochs |
| Loss per epoch | 0.178 → 0.137 → 0.105 (3 epochs, final) |
| Held-out accuracy | 94.4% |
| Export | `torch.onnx.export` (external weights) + `joblib.dump(vectorizer)` |

## Model details

| | |
|---|---|
| Architecture | TF-IDF (30,000 features) → Gemm fc1 (30000→256) → ReLU → Gemm fc2 (256→1) → Sigmoid |
| Size | 5 KB graph + 30,785,536 bytes external weights (~29 MB) |
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
