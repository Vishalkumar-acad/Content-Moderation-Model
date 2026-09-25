# =====================================================================
# PIXELABS CONTENT MODERATION v3 - DEEP LEARNING (DistilBERT) RETRAIN
# Run on: Kaggle (GPU = T4/P100, Internet = ON)
# =====================================================================
# v2 (TF-IDF) me jo galtiyan hui thin, unke fixes is script me built-in:
#   1. RAM BOMB FIX   : saara evaluation chhote batches me (v2 me poori
#                       val matrix ek saath 23GB maangti thi)
#   2. EARLY STOPPING : val improve nahI ho to khud ruk jaata hai
#                       (v2 me 12 epochs me se 8 waste hue the)
#   3. OVERSAMPLE CAP : max x20 (v2 me x4875 ho gaya tha -> 0.8 FP zone)
#   4. DIVERSE DATA   : 60 naye benign garbage/trash sentences + error
#                       additions - generalization ke liye
#   5. HELD-OUT BATTERY: battery cases training me NAHI - asli test
#                       (v2 me battery me trained cases the)
#   6. HONEST TIME    : sizes + ETA pehle print, baad me surprise NahI
#   7. ONNX PARITY    : export ke baad ONNX vs torch scores compare
#                       (export bug chupke NAHI jaayega)
#
# TIME (approx, T4 GPU): data+encode ~15 min | per epoch ~45-75 min
#                        total (early stop ke saath) ~2-3 ghante
#
# KAGGLE SETUP:
#   1. GPU ON, Internet ON
#   2. kaggle-v2-additions.csv wala dataset attached rahe (jo v2 me
#      attach kiya tha - wahi rahega to best; na mile to script
#      warning de kar aage badhegi)
#   3. Ye pura code EK cell me paste karo -> Run
#
# OUTPUT: distilbert_moderation_v3.onnx (fp32)
#         distilbert_moderation_v3_int8.onnx (chhota, ~4x)
#         tokenizer/ (tokenizer.json waghera)
#         In dono/tino ko download karke mujhe bhejo - main deploy karunga.
# =====================================================================

import subprocess, sys, os, glob, csv, random, time

def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

try:
    import transformers  # noqa
except ImportError:
    pip("transformers")
try:
    import datasets  # noqa
except ImportError:
    pip("datasets")
try:
    import sklearn  # noqa
except ImportError:
    pip("scikit-learn")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN     = 128
EPOCHS      = 3        # early stopping khud rok dega
PATIENCE    = 1        # itne epoch tak val improve NahI to stop
BATCH       = 64
LR          = 2e-5
BENIGN_TOTAL = 300_000  # + saara toxic (~140k) = ~440k rows/epoch
ADD_CAP      = 20       # oversample cap (sab tail rows ke liye)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
if device.type != "cuda":
    raise SystemExit("GPU NahI mila! Kaggle settings me Accelerator = GPU ON karo.")

# ---------------------------------------------------------------------
# SECTION 1: DATA
# ---------------------------------------------------------------------
print("=" * 62)
print("SECTION 1: Data load")
from datasets import load_dataset

ds = None
for name in ("Setfit/toxic-conversations", "SetFit/toxic_conversations",
             "getfit/toxic_conversations"):
    try:
        ds = load_dataset(name, split="train")
        print(f"  original dataset: {name} ({len(ds)} rows)")
        break
    except Exception:
        continue
if ds is None:
    raise SystemExit("Dataset load NahI hua - Internet ON hai?")

texts_all = [t.strip() for t in ds["text"] if t and t.strip()]
labels_all = [int(l) for t, l in zip(ds["text"], ds["label"]) if t and t.strip()]
del ds

toxic_idx = [i for i, l in enumerate(labels_all) if l == 1]
benign_idx = [i for i, l in enumerate(labels_all) if l == 0]
print(f"  total: {len(texts_all)} (toxic {len(toxic_idx)}, benign {len(benign_idx)})")

# stratified subsample: SAARA toxic + BENIGN_TOTAL benign
rng = np.random.default_rng(SEED)
benign_pick = rng.choice(benign_idx, size=min(BENIGN_TOTAL, len(benign_idx)),
                          replace=False)
texts = [texts_all[i] for i in toxic_idx] + [texts_all[i] for i in benign_pick]
labels = [1] * len(toxic_idx) + [0] * len(benign_pick)
del texts_all, labels_all

# --- error additions (kaggle-v2-additions.csv) - dedup + CAP x20 ---
add_path = None
for p in glob.glob("/kaggle/input/**/kaggle-v2-additions.csv", recursive=True) \
          + ["kaggle-v2-additions.csv"]:
    if os.path.exists(p):
        add_path = p
        break
n_add_in, n_add_kept = 0, 0
tail_start = len(texts)   # iske baad jo judenge wahi oversample honge
if add_path:
    pool = set(t.lower() for t in texts)
    with open(add_path, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2 and row[0].strip():
                n_add_in += 1
                t = row[0].strip()
                if t.lower() not in pool:          # dedup vs train data
                    pool.add(t.lower())
                    texts.append(t)
                    labels.append(int(row[1]))
                    n_add_kept += 1
    print(f"  additions: {n_add_in} me se {n_add_kept} unique mile"
          f" (baaki original data me pehle se the) -> x{ADD_CAP} repeat honge")
else:
    print("  [WARN] kaggle-v2-additions.csv NahI mili - bina uske chal raha hoon")

# --- embedded DIVERSE benign garbage set (naya - generalization fix) ---
BENIGN_GARBAGE = [
    "I take out the garbage every morning before work",
    "The garbage truck comes every Tuesday in our street",
    "He works as a garbage collector and loves his job",
    "Please remember to empty the trash before you leave",
    "Put the banana peel in the trash, not on the ground",
    "My trash can was knocked over by the wind last night",
    "Volunteers picked up garbage from the beach on Sunday",
    "The garbage disposal in our kitchen is broken again",
    "We sorted the garbage into recycling and compost bins",
    "The city increased garbage collection fees this year",
    "She carried the trash bag outside in the rain",
    "Trash pickup in our neighborhood is on Fridays",
    "The garbage can smells bad, wash it this weekend",
    "Do not litter, use the trash bin next to the door",
    "My phone battery is trash these days, I need a new one",
    "This old laptop is trash, it cannot even open a browser",
    "The movie was trash but the soundtrack was amazing",
    "Honestly this phone camera quality is trash compared to last year",
    "Fast food wrappers filled the trash bin after the party",
    "He forgot to take the trash out and it overflowed",
    "The park installed new trash bins near the benches",
    "Recycling garbage properly helps the environment a lot",
    "Our office garbage gets cleared every evening",
    "The kids helped collect trash along the hiking trail",
    "That restaurant's food is trash, never going there again",
    "The trash chute in our building is always jammed",
    "Garbage collection was delayed due to the holiday",
    "Please bag your garbage tightly to avoid spills",
    "The smell of garbage was strong near the dumpster",
    "Old furniture collection is different from garbage pickup",
    "My internet connection is trash lately, nothing loads",
    "Battery life on this watch is trash, dies by noon",
    "The trash bags tore open on the way to the curb",
    "Sorting garbage correctly takes a bit of practice",
    "The city fined people for throwing garbage on streets",
    "Our garbage bin attracts raccoons every night",
    "Take the kitchen trash out when you leave, please",
    "The garbage strike made the whole street stink",
    "Plastic garbage in the ocean harms marine life",
    "This printer is trash, it jams on every third page",
    "He does garbage duty at the community center",
    "Composting reduced our household garbage by half",
    "The trash compactor at the mall is really loud",
    "Do not throw garbage from the car window",
    "Street cleaners collect garbage at dawn here",
    "That TV show's writing is trash but I still watch it",
    "The overflow of trash at the concert was disappointing",
    "My old headphones were trash so I bought new ones",
    "Garbage cans must be placed on the curb by 6 am",
    "The raccoon knocked over our garbage again",
    "This hotel's wifi is trash, barely one bar",
    "We need better garbage management in this city",
    "The trash island in the Pacific keeps growing",
    "Disposable cups fill the trash after every meeting",
    "She tripped over a trash bag in the dark hallway",
    "Our garbage man comes at five in the morning",
    "The new trash policy requires clear bags only",
    "Battery recycling bin is next to the garbage bin",
    "That pizza place's crust is trash, too soggy",
]
TOXIC_ANCHORS = [
    "You are garbage",
    "You are a garbage human being",
    "Your entire family is trash",
    "Keep talking garbage and see what happens",
    "Your writing is garbage and so are you",
    "You are a worthless piece of trash",
    "Everything you make is garbage",
    "You smell like garbage and act like it too",
    "Your opinions are trash and you should stop sharing them",
    "You talk garbage all day long",
]
n_emb = 0
pool = set(t.lower() for t in texts)
for t in BENIGN_GARBAGE:
    if t.lower() not in pool:
        pool.add(t.lower()); texts.append(t); labels.append(0); n_emb += 1
for t in TOXIC_ANCHORS:
    if t.lower() not in pool:
        pool.add(t.lower()); texts.append(t); labels.append(1); n_emb += 1
print(f"  embedded diverse set: {n_emb} unique add hue (60 benign garbage + "
      f"10 toxic anchors me se) -> x{ADD_CAP} repeat honge")

# additions + embedded set ko oversample (CAP ke saath - v2 wali x4875 NahI)
n_tail = len(texts) - tail_start
texts = texts + texts[tail_start:] * (ADD_CAP - 1)
labels = labels + labels[tail_start:] * (ADD_CAP - 1)
print(f"  oversample: {n_tail} tail rows -> x{ADD_CAP} total "
      f"(+{n_tail*(ADD_CAP-1)} copies)")

# ---------------------------------------------------------------------
# SECTION 2: BATTERY (held-out) - in cases ko training se HATAO
# ---------------------------------------------------------------------
BATTERY = [
    # benign garbage (kuch embedded jaise, kuch NAYE)
    ("Please put the garbage in the trash", False),
    ("Please take out the garbage", False),
    ("Take the garbage out to the curb before sunrise", False),
    ("Sorting garbage and recycling is everyone's job", False),
    ("He works as a garbage collector in our city", False),
    ("The garbage truck comes on Monday", False),
    ("Trash pickup in our neighborhood is on Fridays", False),
    ("My phone battery is trash", False),
    ("The trash can is full, someone empty it", False),
    ("Garbage collection is on Tuesdays this week", False),
    # benign clean
    ("I love this post, very well written", False),
    ("I could kill for a cup of coffee right now", False),
    ("Her presentation killed it, everyone loved it", False),
    ("He was niggardly with his praise", False),
    # toxic
    ("You are totally trash", True),
    ("You are trash", True),
    ("You are a worthless piece of garbage", True),
    ("Nobody likes you, just leave", True),
    ("Shut up you idiot", True),
    ("Assholes", True),
    ("NIGGER", True),
    ("You are such a scumbag", True),
    ("Stop being an assclown", True),
    ("That is complete horseshit", True),
]
bat_set = set(t.lower() for t, _ in BATTERY)
keep = [i for i, t in enumerate(texts) if t.lower() not in bat_set]
texts = [texts[i] for i in keep]
labels = [labels[i] for i in keep]
print(f"  battery ke {len(BATTERY)} cases training se hata diye (asli generalization test)")

print(f"\n  FINAL TRAIN POOL: {len(texts)} rows "
      f"(toxic {sum(labels)}, benign {len(labels)-sum(labels)})")

# ---------------------------------------------------------------------
# SECTION 3: Tokenize + split
# ---------------------------------------------------------------------
print("=" * 62)
print("SECTION 3: Tokenize")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode(texts_list):
    ids, masks = [], []
    B = 2000
    for s in range(0, len(texts_list), B):
        enc = tok(texts_list[s:s+B], truncation=True, max_length=MAX_LEN,
                  padding="max_length")
        ids.extend(enc["input_ids"])
        masks.extend(enc["attention_mask"])
    return (np.array(ids, dtype=np.int64), np.array(masks, dtype=np.int64))

input_ids, attn = encode(texts)
y = np.array(labels, dtype=np.float32)
print(f"  encoded {len(texts)} rows in {time.time()-t0:.0f}s")

Xtr_i, Xva_i, Xtr_m, Xva_m, ytr, yva = train_test_split(
    input_ids, attn, y, test_size=0.02, random_state=SEED, stratify=y)
print(f"  train: {len(ytr)}  val: {len(yva)}")
del input_ids, attn, texts, labels

class TDataset(Dataset):
    def __init__(self, ids, masks, ys):
        self.ids, self.masks, self.ys = ids, masks, ys
    def __len__(self):
        return len(self.ys)
    def __getitem__(self, i):
        return (torch.from_numpy(self.ids[i]), torch.from_numpy(self.masks[i]),
                torch.tensor(self.ys[i]))

train_dl = DataLoader(TDataset(Xtr_i, Xtr_m, ytr), batch_size=BATCH,
                      shuffle=True, num_workers=2, pin_memory=True)

# ---------------------------------------------------------------------
# SECTION 4: TRAIN (fp16 + early stopping + progress + best checkpoint)
# ---------------------------------------------------------------------
print("=" * 62)
print("SECTION 4: Training (fp16, early stopping ON)")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
try:
    scaler = torch.amp.GradScaler("cuda")
except Exception:
    scaler = torch.cuda.amp.GradScaler()   # purane torch versions

def predict_batched(model, ids, masks, bs=256):
    """CHHOTE batches me predict - RAM bomb NahI (v2 wali galti fix)."""
    model.eval()
    out = np.zeros(len(ids), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(ids), bs):
            i = torch.from_numpy(ids[s:s+bs]).to(device)
            m = torch.from_numpy(masks[s:s+bs]).to(device)
            with torch.autocast("cuda"):
                p = model(input_ids=i, attention_mask=m).logits
            out[s:s+bs] = torch.sigmoid(p.float()).cpu().numpy().ravel()
    return out

best_auc, best_state, no_imp = -1.0, None, 0
for epoch in range(EPOCHS):
    model.train()
    t0 = time.time()
    tot, nb = 0.0, 0
    for b, (bi, bm, by) in enumerate(train_dl):
        bi, bm, by = bi.to(device), bm.to(device), by.unsqueeze(1).to(device)
        optimizer.zero_grad()
        with torch.autocast("cuda"):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(input_ids=bi, attention_mask=bm).logits, by)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tot += loss.item(); nb += 1
        if b % 500 == 0 and b > 0:
            el = time.time() - t0
            done = b / len(train_dl)
            eta = el / done * (1 - done) / 60
            print(f"    epoch {epoch+1}: batch {b}/{len(train_dl)} "
                  f"loss {tot/nb:.4f} | elapsed {el/60:.0f}m | ETA {eta:.0f}m",
                  flush=True)
    pv = predict_batched(model, Xva_i, Xva_m)
    auc = roc_auc_score(yva, pv)
    print(f"  epoch {epoch+1} DONE in {(time.time()-t0)/60:.0f}m | "
          f"loss {tot/max(nb,1):.4f} | val_AUC {auc:.4f}", flush=True)
    if auc > best_auc:
        best_auc, no_imp = auc, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print(f"  early stop: {PATIENCE} epoch se improvement NahI - ruk gaye")
            break

model.load_state_dict(best_state)
model.eval()
pv = predict_batched(model, Xva_i, Xva_m)
acc = accuracy_score(yva, (pv >= 0.5).astype(int))
fp = int(((pv >= 0.5) & (yva == 0)).sum()); fn = int(((pv < 0.5) & (yva == 1)).sum())
print(f"\n  BEST: val_AUC {best_auc:.4f} | val_acc {acc*100:.2f}% | "
      f"FP {fp} ({fp/len(yva)*100:.2f}%) | FN {fn} ({fn/len(yva)*100:.2f}%)")

# ---------------------------------------------------------------------
# SECTION 5: BATTERY (deploy gate) - torch se
# ---------------------------------------------------------------------
print("=" * 62)
print("SECTION 5: Battery (held-out cases, asli test)")
bat_texts = [t for t, _ in BATTERY]
bi_ids, bi_masks = encode(bat_texts)
pb = predict_batched(model, bi_ids, bi_masks)
npass = 0
for (t, exp), sc in zip(BATTERY, pb):
    ok = (sc >= 0.5) == exp
    npass += ok
    print(f"  {'PASS' if ok else 'FAIL'} [{sc:.4f}] {t[:58]}")
print(f"  RESULT: {npass}/{len(BATTERY)}")
if npass < len(BATTERY) - 2:
    print("  !! 2 se zyada FAIL - deploy mat karo, mujhe output bhejo")

# ---------------------------------------------------------------------
# SECTION 6: ONNX EXPORT (fp32 + int8) + TOKENIZER + PARITY CHECK
# ---------------------------------------------------------------------
print("=" * 62)
print("SECTION 6: ONNX export + parity check")

class OnnxWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids, attention_mask):
        return torch.sigmoid(self.m(input_ids=input_ids,
                                    attention_mask=attention_mask).logits)

wrap = OnnxWrap(model).eval()
dummy_i = torch.from_numpy(bi_ids[:2]).to(device)
dummy_m = torch.from_numpy(bi_masks[:2]).to(device)
_exp_kw = dict(
    input_names=["input_ids", "attention_mask"], output_names=["score"],
    dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                  "score": {0: "batch"}},
    opset_version=17,
)
try:
    # classic exporter - int8 quantization isi ke saath kaam karti hai
    torch.onnx.export(wrap, (dummy_i, dummy_m), "distilbert_moderation_v3.onnx",
                      **_exp_kw, dynamo=False)
except TypeError:
    # purana torch jisme dynamo param NahI hai
    torch.onnx.export(wrap, (dummy_i, dummy_m), "distilbert_moderation_v3.onnx",
                      **_exp_kw)

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic("distilbert_moderation_v3.onnx",
                     "distilbert_moderation_v3_int8.onnx",
                     weight_type=QuantType.QInt8)
    print("  int8 quantize: OK")
except Exception as e:
    print(f"  [WARN] int8 quantize fail ({e}) - fp32 hi use hoga")

tok.save_pretrained("tokenizer")

# PARITY: ONNX vs torch (export bug chupke NahI jayega - v2 lesson)
import onnxruntime as ort
sess = ort.InferenceSession("distilbert_moderation_v3.onnx",
                            providers=["CPUExecutionProvider"])
onnx_scores = []
B = 64
for s in range(0, len(bat_texts), B):
    r = sess.run(["score"], {
        "input_ids": bi_ids[s:s+B],
        "attention_mask": bi_masks[s:s+B],
    })[0].ravel()
    onnx_scores.extend(float(x) for x in r)
maxdiff = max(abs(a - b) for a, b in zip(pb, onnx_scores))
print(f"  ONNX vs torch parity: max diff {maxdiff:.5f} "
      f"({'OK' if maxdiff < 0.01 else 'PROBLEM - mat deploy'})")

for f in ("distilbert_moderation_v3.onnx", "distilbert_moderation_v3_int8.onnx"):
    if os.path.exists(f):
        print(f"  {f}: {os.path.getsize(f)/1e6:.1f} MB")
print("  tokenizer/: tokenizer.json + config files saved")

print()
print("HO GAYA! Download karo: distilbert_moderation_v3.onnx (+ int8), "
      "tokenizer/ folder")
print("Mujhe upload karo - main naya main.py (transformer pipeline) likh kar")
print("deploy kar dunga - wahi sha256 fail-closed flow.")
