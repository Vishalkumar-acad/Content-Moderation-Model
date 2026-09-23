# Content-Moderation-Model
A highly accurate (94.4%) and lightweight Deep Learning model for real-time content and comment moderation. Trained on 1.75 million toxic conversations using PyTorch and TF-IDF, exported in ONNX format for fast inference on Node.js/Nitro backends.

## Serving (Render.com)

The repo ships a FastAPI server (`main.py`) + `requirements.txt` + `render.yaml`:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Endpoints

- `GET /` — health + model status
- `POST /moderate` — body: `{"text": "..."}` → `{"toxic": true|false, "score": 0.0–1.0, "label": "toxic"|"neutral"}`

## Model weights (important)

The ONNX graph (`content_moderation_gpu.onnx`) stores its weights in an external
file `content_moderation_gpu.onnx.data` — **30,785,536 bytes** (fc1.bias @ 0,
fc2.weight @ 1024, fc1.weight @ 65536). The file originally committed to the
git tree was truncated at exactly 4 MiB, so the complete file is published as a
[release asset](https://github.com/Vishalkumar-acad/Content-Moderation-Model/releases/tag/model-v1).

`main.py` handles this automatically: if the local `.onnx.data` is missing or
too small, it downloads the full file from the release in a background thread
(size + sha256 verified) and then loads the model. The web server binds its port
immediately, so the Render health check passes even during the download.

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI inference server (auto-downloads weights) |
| `content_moderation_gpu.onnx` | ONNX model graph (5 KB) |
| `content_moderation_gpu.onnx.data` | External weights — via the model-v1 release asset |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer (30,000 features, scikit-learn 1.6.1) |
| `render.yaml` | Render.com deploy config |

## Architecture

TF-IDF (30,000 features) → Gemm(fc1: 30000→256) → ReLU → Gemm(fc2: 256→1) → Sigmoid → probability. The graph output is already a probability — no extra sigmoid needed at inference time.
