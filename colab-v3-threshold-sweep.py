# =====================================================================
# v3 THRESHOLD SWEEP - SAME 20,000 comments (seed 20260924)
# Run on: Colab ya Kaggle  |  Time: ~15-20 minute
# =====================================================================
# Pichhli baar sweep cell NameError se fail hui thi (sc3 variable chala
# gaya tha). Ye script SELF-CONTAINED hai: sab kuch khud download karke
# chalati hai, aur results ko FILE me bhi save karti hai (sc3.npy) taaki
# dobara kabhi loss na ho.
#
# ANCHOR CHECK: 0.5 wali row me FP=616, FN=16 hona CHAHIYE (wahi jo
# tumhare comparison run me aaya tha). Agar alag aaya to data/seed
# mismatch hai - mujhe turant batao.
# =====================================================================

import subprocess, sys

def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip("fastapi", "uvicorn", "onnxruntime", "tokenizers", "numpy", "datasets")

import hashlib
import time
import urllib.request
import numpy as np

RAW = "https://raw.githubusercontent.com/Vishalkumar-acad/Content-Moderation-Model/main/"
REL = "https://github.com/Vishalkumar-acad/Content-Moderation-Model/releases/download/model-v3/"

print("=" * 62)
print("SECTION 1: Production v3 files download (sha-verified)")
print("=" * 62)
urllib.request.urlretrieve(RAW + "main.py", "main.py")
_got = hashlib.sha256(open("main.py", "rb").read()).hexdigest()
if _got != "9402bdf417dc026b22a591067917fd1c7f1dbf641e9b3e0a4c745cb232c536bd":
    raise SystemExit(f"main.py sha mismatch! got {_got} - 2 min ruk ke dobara chalao")
print("  main.py v1.5.0 verified")
for f in ("distilbert_moderation_v3_int8.onnx", "tokenizer.json", "tokenizer_config.json"):
    print(f"  {f} download...")
    urllib.request.urlretrieve(REL + f, f)

import main as M
assert M.model_status == "ok", "model load fail"
print(f"  ready: {M.app.version}")

# ------------------------------------------------------------------
print("=" * 62)
print("SECTION 2: WAHI 20,000 comments (seed 20260924 - v2/v3 wale)")
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
print("  (expected: 20000 / toxic 502 / benign 19498 - agar alag hai to batana)")

# ------------------------------------------------------------------
print("=" * 62)
print("SECTION 3: v3 full pipeline scoring (teeno layers)")
print("=" * 62)

def full_pipeline_scores(texts):
    scores = np.zeros(len(texts)); need = []
    for i, t in enumerate(texts):
        if M._PROFANITY_RE.search(t):
            scores[i] = 1.0
        elif any(r.search(t) for r in M._HARASSMENT_RES):
            scores[i] = 1.0
        else:
            need.append(i)
    B = 256
    for s in range(0, len(need), B):
        idx = need[s:s + B]
        encs = M.tokenizer.encode_batch([texts[i] for i in idx])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        masks = np.array([e.attention_mask for e in encs], dtype=np.int64)
        p = M.session.run(["score"], {"input_ids": ids, "attention_mask": masks})[0].ravel()
        scores[idx] = p
        if (s // B) % 10 == 0:
            print(f"    ...{s + B}/{len(need)}", flush=True)
    return scores

t0 = time.time()
sc3 = full_pipeline_scores(texts)
print(f"  done ({time.time()-t0:.0f}s)")

# SAVE - ab ye kabhi loss NahI honge
np.save("sc3.npy", sc3)
np.save("labels.npy", labels)
print("  saved: sc3.npy + labels.npy (dobara chahiye to np.load se)")

# ------------------------------------------------------------------
print()
print("=" * 62)
print("THRESHOLD SWEEP (v3) - ASLI NUMBERS")
print("=" * 62)
n_tox = int(labels.sum()); n_ben = len(labels) - n_tox
print(f"{'thr':<6}{'FP':>6} {'FP%':>7} {'FN':>5} {'FN%':>7}   (benign={n_ben}, toxic={n_tox})")
for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
    fp = int(((sc3 >= t) & (labels == 0)).sum())
    fn = int(((sc3 < t) & (labels == 1)).sum())
    print(f"{t:<6}{fp:>6} {fp/n_ben*100:>6.2f}% {fn:>5} {fn/n_tox*100:>6.2f}%")
print()
print("ANCHOR CHECK: 0.5 wali row me FP=616, FN=16 hona chahiye")
print("(wahi jo comparison run me aaya tha - match = sab sahi hai)")
print()
print("v2 reference (fixed): FP=127 (0.64%), FN=83 (0.41%) @ 0.5")
