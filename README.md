# Content-Moderation-Model
A highly accurate (94.4%) and lightweight Deep Learning model for real-time content and comment moderation. Trained on 1.75 million toxic conversations using PyTorch and TF-IDF, exported in ONNX format for fast inference on Node.js/Nitro backends.

## Serving (Render.com)

The repo ships a FastAPI server (`main.py`) + `requirements.txt` + `render.yaml`:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Endpoints

- `GET /` — health + model status
- `POST /moderate` — body: `{"text": "..."}` → `{"toxic": true|false, "score": 0.0–1.0, "label": "toxic"|"neutral"}`

### Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI inference server |
| `content_moderation_gpu.onnx` | ONNX model graph (5 KB) |
| `content_moderation_gpu.onnx.data` | External weights — **must be exactly 30,785,536 bytes** (see below) |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer (30,000 features, scikit-learn 1.6.1) |

### Important: `.onnx.data` size

The ONNX graph stores its weights in `content_moderation_gpu.onnx.data` at fixed
offsets (`fc1.bias` @ 0, `fc2.weight` @ 1024, `fc1.weight` @ 65536 with a length
of 30,720,000). The file therefore must be at least **30,785,536 bytes
(~29.4 MB)**. If the file on GitHub is smaller (e.g. exactly 4,194,304 bytes),
it was truncated during upload and the model cannot load — re-upload the full
file. `GET /` reports the exact problem if the file is bad.
