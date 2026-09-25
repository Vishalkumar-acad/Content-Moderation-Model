# =====================================================================
# PIXELABS v3 (DistilBERT) - FULL PIPELINE ACCURACY - SAME 20,000 DATA
# jis data pe v2 test hua tha (google/civil_comments, seed 20260924)
# Run on: Colab ya Kaggle (CPU theek hai)  |  Time: ~15-20 minute
# =====================================================================
# Ye script PRODUCTION wala POORA v3 pipeline test karti hai:
#   1. Profanity wordlist (nigger, horseshit, assclown, scumbag, ...)
#   2. Harassment patterns ("nobody likes you", ...)
#   3. DistilBERT v3 int8 model (ONNX)
#
# Sab kuch DIRECT GitHub se download hota hai - main.py BHI wahi jo
# Render par live hai (exact production code, sha256-verified).
# Fir WAHI 20,000 comments (civil_comments, WAHI seed 20260924 jo
# v2 test me tha) par accuracy nikalti hai - direct v2-vs-v3 compare.
# =====================================================================

import subprocess, sys

def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip("fastapi", "uvicorn", "onnxruntime", "tokenizers", "numpy", "datasets", "scikit-learn")

import hashlib
import time
import urllib.request
import numpy as np

RAW = "https://raw.githubusercontent.com/Vishalkumar-acad/Content-Moderation-Model/main/"
REL = "https://github.com/Vishalkumar-acad/Content-Moderation-Model/releases/download/model-v3/"

print("=" * 62)
print("SECTION 1: Production files download (direct GitHub se)")
print("=" * 62)
# main.py = EXACT wahi code jo Render par chal raha hai (v1.5.0)
urllib.request.urlretrieve(RAW + "main.py", "main.py")
_got = hashlib.sha256(open("main.py", "rb").read()).hexdigest()
_EXP = "9402bdf417dc026b22a591067917fd1c7f1dbf641e9b3e0a4c745cb232c536bd"
if _got != _EXP:
    raise SystemExit(f"main.py sha256 mismatch! got {_got} - RAW url cache me purani file hai, 2 minute ruk ke dobara chalao")
print("  main.py v1.5.0 verified (sha256 ok)")

# model files release se (67MB) - main.py KHUD bhi sha256 verify karega
print("  model download ho raha hai (67MB)...")
for f in ("distilbert_moderation_v3_int8.onnx", "tokenizer.json", "tokenizer_config.json"):
    urllib.request.urlretrieve(REL + f, f)

# import main -> fail-closed: galat file hui to error, kabhi galat model NahI chalega
import main as M

print(f"  server version : {M.app.version}")
print(f"  model_status   : {M.model_status}")
assert M.model_status == "ok" and M.tokenizer is not None, "model load fail!"
print("  sab load ho gaya - ye EXACT production v3 pipeline hai")

# ------------------------------------------------------------------
print("=" * 62)
print("SECTION 2: Test data - WAHI 20,000 comments (v2 wala seed)")
print("=" * 62)
from datasets import load_dataset

N = 20000
SEED = 20260924  # v2 test wala SAME seed -> same 20,000 comments
ds = load_dataset("google/civil_comments", split="train", streaming=True)
ds = ds.shuffle(seed=SEED, buffer_size=100_000)
texts, labels = [], []
for row in ds:
    t = row["text"].strip()
    if not t:
        continue
    if row["toxicity"] >= 0.7:
        lab = 1          # clearly toxic
    elif row["toxicity"] <= 0.3:
        lab = 0          # clearly benign
    else:
        continue         # ambiguous skip
    texts.append(t)
    labels.append(lab)
    if len(texts) >= N:
        break
labels = np.array(labels)
print(f"  comments: {len(texts)} (toxic {labels.sum()}, benign {len(labels)-labels.sum()})")

# ------------------------------------------------------------------
print("=" * 62)
print("SECTION 3: v3 full pipeline se scoring (wordlist + harassment + DistilBERT)")
print("=" * 62)

def full_pipeline_scores(texts):
    """Production /moderate ka EXACT logic - teeno layers."""
    scores = np.zeros(len(texts))
    layer = [None] * len(texts)
    need_model = []
    for i, t in enumerate(texts):
        if M._PROFANITY_RE.search(t):
            scores[i] = 1.0
            layer[i] = "profanity_list"
        elif any(r.search(t) for r in M._HARASSMENT_RES):
            scores[i] = 1.0
            layer[i] = "harassment_pattern"
        else:
            need_model.append(i)
    # bacha hua DistilBERT se (batch me - fast)
    B = 256
    for s in range(0, len(need_model), B):
        idx = need_model[s:s + B]
        encs = M.tokenizer.encode_batch([texts[i] for i in idx])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        masks = np.array([e.attention_mask for e in encs], dtype=np.int64)
        p = M.session.run(["score"], {"input_ids": ids, "attention_mask": masks})[0].ravel()
        scores[idx] = p
        for i in idx:
            layer[i] = "model"
        if (s // B) % 10 == 0:
            print(f"    ...{s + B}/{len(need_model)} model se ho gaye", flush=True)
    return scores, layer

t0 = time.time()
scores, layer = full_pipeline_scores(texts)
print(f"  {len(texts)} comments score ho gaye ({time.time()-t0:.0f}s)")

pred = (scores >= 0.5).astype(int)
acc = float((pred == labels).mean())
tp = int(((pred == 1) & (labels == 1)).sum())
tn = int(((pred == 0) & (labels == 0)).sum())
fp = int(((pred == 1) & (labels == 0)).sum())
fn = int(((pred == 0) & (labels == 1)).sum())

from collections import Counter
lay = Counter(layer[i] for i in range(len(texts)) if pred[i] == 1)

print()
print("=" * 62)
print("OVERALL ACCURACY - v3 (wordlist + harassment + DistilBERT)")
print("=" * 62)
print(f"  ACCURACY  : {acc*100:.2f}%   ({(pred==labels).sum()}/{len(labels)})")
print(f"  Correct  : {tp+tn}   (toxic pakde {tp}, benign sahi {tn})")
print(f"  FALSE POS: {fp}  ({fp/len(labels)*100:.2f}% - innocent ko toxic bola)")
print(f"  FALSE NEG: {fn}  ({fn/len(labels)*100:.2f}% - toxic chhoota)")
print(f"  Toxic catches kis layer se: {dict(lay)}")

try:
    from sklearn.metrics import roc_auc_score
    print(f"  AUC (pipeline scores): {roc_auc_score(labels, scores):.4f}")
except Exception as e:
    print("  (AUC skip:", e, ")")

# ------------------------------------------------------------------
print()
print("=" * 62)
print("KNOWN-CASE BATTERY (chhota sanity check)")
print("=" * 62)
BATTERY = [
    ("Please put the garbage in the trash", 0),
    ("He works as a garbage collector in our city", 0),
    ("My phone battery is trash", 0),
    ("I could kill for a cup of coffee right now", 0),
    ("Her presentation killed it, everyone loved it", 0),
    ("You are totally trash", 1),
    ("You are a worthless piece of garbage", 1),
    ("Nobody likes you, just leave", 1),
    ("NIGGER", 1),
    ("That is complete horseshit", 1),
    ("You are such a scumbag", 1),
    ("Stop being an assclown", 1),
    ("He was niggardly with his praise", 0),
]
s2, l2 = full_pipeline_scores([t for t, _ in BATTERY])
ok = 0
for (t, exp), sc, ly in zip(BATTERY, s2, l2):
    good = (sc >= 0.5) == bool(exp)
    ok += good
    print(f"  {'PASS' if good else 'FAIL'} [{sc:.4f}] ({ly}) {t[:52]}")
print(f"  BATTERY: {ok}/{len(BATTERY)} pass")

print()
print("HO GAYA - ye numbers v2 ke numbers se compare karo (same 20k data):")
print("v2 me kya aaya tha usko yaad rakhna/likh lena - FP/FN me zyada")
print("safalta dikhna chahiye kyunki DistilBERT context samajhta hai.")
