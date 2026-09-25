# =====================================================================
# V2 vs V3 COMPARE - SAME 20,000 comments pe dono models, ek hi run me
# Run on: Colab ya Kaggle (CPU theek hai)  |  Time: ~25-35 minute
# =====================================================================
# v2 ka detail lost ho gaya tha - isliye ye script v2 ko wapas GitHub se
# uthati hai (model-v2 release + purana main.py v1.4.1, sha-verified) aur
# v3 ke saath SAME data pe chala kar side-by-side table deti hai.
# Data: wahi civil_comments, wahi seed 20260924 - 100% fair comparison.
# =====================================================================

import subprocess, sys

def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip("fastapi", "uvicorn", "onnxruntime", "tokenizers", "numpy",
    "datasets", "scikit-learn==1.6.1", "joblib")

import hashlib
import time
import urllib.request
import numpy as np

RAW = "https://raw.githubusercontent.com/Vishalkumar-acad/Content-Moderation-Model/main/"
V2RAW = "https://raw.githubusercontent.com/Vishalkumar-acad/Content-Moderation-Model/9db847dbe4f933aa07d396c028bf86d73aa55be7/"
REL2 = "https://github.com/Vishalkumar-acad/Content-Moderation-Model/releases/download/model-v2/"
REL3 = "https://github.com/Vishalkumar-acad/Content-Moderation-Model/releases/download/model-v3/"

print("=" * 62)
print("SECTION 1: Dono models download (v2 + v3, direct GitHub se)")
print("=" * 62)

# ---- v3: main.py v1.5.0 (sha-verified) + model-v3 assets
urllib.request.urlretrieve(RAW + "main.py", "main.py")
_got = hashlib.sha256(open("main.py", "rb").read()).hexdigest()
if _got != "9402bdf417dc026b22a591067917fd1c7f1dbf641e9b3e0a4c745cb232c536bd":
    raise SystemExit(f"main.py v3 sha mismatch! got {_got} - 2 min ruk ke dobara chalao")
print("  v3 main.py (v1.5.0) verified")
for f in ("distilbert_moderation_v3_int8.onnx", "tokenizer.json", "tokenizer_config.json"):
    print(f"  v3 {f} download...")
    urllib.request.urlretrieve(REL3 + f, f)

# ---- v2: purana main.py v1.4.1 (pinned commit, sha-verified) + model-v2
urllib.request.urlretrieve(V2RAW + "main.py", "main_v2.py")
_got = hashlib.sha256(open("main_v2.py", "rb").read()).hexdigest()
if _got != "b6f2674a3bb024c64d5c5ba70ce9991ab83ef738ae19a5f6b06d8dccd21dde5c":
    raise SystemExit(f"main_v2.py sha mismatch! got {_got}")
print("  v2 main_v2.py (v1.4.1) verified")
urllib.request.urlretrieve(V2RAW + "content_moderation_gpu.onnx", "content_moderation_gpu.onnx")
for f in ("content_moderation_gpu.onnx.data", "tfidf_vectorizer.pkl"):
    print(f"  v2 {f} download (31MB)...")
    urllib.request.urlretrieve(REL2 + f, f)

import main as M3      # v3 (DistilBERT)
import main_v2 as M2   # v2 (TF-IDF)

assert M3.model_status == "ok", "v3 load fail"
assert M2.model_status == "ok", "v2 load fail"
print(f"  v3 ready: {M3.app.version} | v2 ready: {M2.app.version}")

# ------------------------------------------------------------------
print("=" * 62)
print("SECTION 2: SAME 20,000 comments (v3 test wala seed 20260924)")
print("=" * 62)
from datasets import load_dataset

N = 20000
SEED = 20260924
ds = load_dataset("google/civil_comments", split="train", streaming=True)
ds = ds.shuffle(seed=SEED, buffer_size=100_000)
texts, labels = [], []
for row in ds:
    t = row["text"].strip()
    if not t:
        continue
    if row["toxicity"] >= 0.7:
        lab = 1
    elif row["toxicity"] <= 0.3:
        lab = 0
    else:
        continue
    texts.append(t)
    labels.append(lab)
    if len(texts) >= N:
        break
labels = np.array(labels)
print(f"  comments: {len(texts)} (toxic {labels.sum()}, benign {len(labels)-labels.sum()})")

# ------------------------------------------------------------------
print("=" * 62)
print("SECTION 3: v2 pipeline scoring (wordlist + harassment + TF-IDF)")
print("=" * 62)

def v2_scores(texts):
    scores = np.zeros(len(texts)); need = []
    for i, t in enumerate(texts):
        if M2._PROFANITY_RE.search(t):
            scores[i] = 1.0
        elif any(r.search(t) for r in M2._HARASSMENT_RES):
            scores[i] = 1.0
        else:
            need.append(i)
    B = 1024
    for s in range(0, len(need), B):
        idx = need[s:s + B]
        X = M2.vectorizer.transform([texts[i] for i in idx]).toarray().astype(np.float32)
        p = M2.session.run(["output"], {"input": X})[0].ravel()
        scores[idx] = p
        if (s // B) % 5 == 0:
            print(f"    ...v2: {s + B}/{len(need)}", flush=True)
    return scores

t0 = time.time()
sc2 = v2_scores(texts)
print(f"  v2 done ({time.time()-t0:.0f}s)")

print("=" * 62)
print("SECTION 4: v3 pipeline scoring (wordlist + harassment + DistilBERT)")
print("=" * 62)

def v3_scores(texts):
    scores = np.zeros(len(texts)); need = []
    for i, t in enumerate(texts):
        if M3._PROFANITY_RE.search(t):
            scores[i] = 1.0
        elif any(r.search(t) for r in M3._HARASSMENT_RES):
            scores[i] = 1.0
        else:
            need.append(i)
    B = 256
    for s in range(0, len(need), B):
        idx = need[s:s + B]
        encs = M3.tokenizer.encode_batch([texts[i] for i in idx])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        masks = np.array([e.attention_mask for e in encs], dtype=np.int64)
        p = M3.session.run(["score"], {"input_ids": ids, "attention_mask": masks})[0].ravel()
        scores[idx] = p
        if (s // B) % 10 == 0:
            print(f"    ...v3: {s + B}/{len(need)}", flush=True)
    return scores

t0 = time.time()
sc3 = v3_scores(texts)
print(f"  v3 done ({time.time()-t0:.0f}s)")

# ------------------------------------------------------------------
def metrics(scores):
    pred = (scores >= 0.5).astype(int)
    acc = float((pred == labels).mean())
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return acc, tp, tn, fp, fn

a2, tp2, tn2, fp2, fn2 = metrics(sc2)
a3, tp3, tn3, fp3, fn3 = metrics(sc3)

try:
    from sklearn.metrics import roc_auc_score
    auc2 = roc_auc_score(labels, sc2)
    auc3 = roc_auc_score(labels, sc3)
except Exception:
    auc2 = auc3 = float("nan")

print()
print("=" * 66)
print("FINAL COMPARISON - SAME 20,000 COMMENTS")
print("=" * 66)
print(f"                          v2 (TF-IDF)      v3 (DistilBERT)")
print(f"  ----------------------------------------------------------")
print(f"  ACCURACY            {a2*100:8.2f}%       {a3*100:8.2f}%")
print(f"  Toxic pakde (TP)    {tp2:8d}         {tp3:8d}")
print(f"  Benign sahi (TN)    {tn2:8d}         {tn3:8d}")
print(f"  FALSE POSITIVE      {fp2:8d} ({fp2/len(labels)*100:.2f}%)    {fp3:8d} ({fp3/len(labels)*100:.2f}%)")
print(f"  FALSE NEGATIVE      {fn2:8d} ({fn2/len(labels)*100:.2f}%)    {fn3:8d} ({fn3/len(labels)*100:.2f}%)")
print(f"  AUC                 {auc2:8.4f}         {auc3:8.4f}")
print()
print("  FP = innocent comment ko toxic bola (user ka dukh)")
print("  FN = toxic comment chhoot gaya (moderation ka dukh)")
print()
print("Iska screenshot bhej dena - main final verdict likh dunga.")
