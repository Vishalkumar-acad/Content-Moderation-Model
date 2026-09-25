# =====================================================================
# PIXELABS v3 - VERIFY + DEPLOY BUNDLE (Kaggle par hi sab saaf karo)
# Run: jis notebook session me model files hain (v3.0 abhi / v3.1 baad me)
# Us notebook ke /kaggle/working me ye files honi chahiye:
#   distilbert_moderation_v3.onnx (fp32)
#   distilbert_moderation_v3_int8.onnx
#   tokenizer/ (tokenizer.json waghera)
# =====================================================================
# KYUN YE SCRIPT:
#   Tumhari bheji int8 file ko maine yahan test kiya - "You are trash"
#   ka score 0.24 aa raha hai (0.98 hona chahiye). Do possibilities:
#   (a) phone/upload transfer me file bigad gayi, ya
#   (b) int8 quantization isi graph par kharaab hai
#   Ye script dono check karegi AUR fix bhi karegi - sab Kaggle par.
# OUTPUT: v3-deploy.zip (int8 + tokenizer) + sha256 print
# =====================================================================

import subprocess, sys, os, glob, hashlib, zipfile

def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

try:
    import onnxruntime  # noqa
except ImportError:
    pip("onnxruntime")
try:
    import onnx  # noqa
except ImportError:
    pip("onnx")

import numpy as np

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------
# files dhoondo
# ---------------------------------------------------------------------
def find_one(name):
    for p in [name, os.path.join("/kaggle/working", name)] + \
             glob.glob(f"/kaggle/input/**/{name}", recursive=True):
        if os.path.exists(p):
            return p
    return None

fp32 = find_one("distilbert_moderation_v3.onnx")
int8 = find_one("distilbert_moderation_v3_int8.onnx")
tokdir = None
for c in ["tokenizer", "/kaggle/working/tokenizer"] + \
         [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/tokenizer/tokenizer.json", recursive=True)]:
    if os.path.isfile(os.path.join(c, "tokenizer.json")):
        tokdir = c
        break
print("=" * 62)
print("FILES:")
print(f"  fp32:     {fp32}")
print(f"  int8:     {int8}")
print(f"  tokenizer: {tokdir}")
if fp32 is None or tokdir is None:
    raise SystemExit("fp32 ya tokenizer NahI mila - model wale session me chalao")

if int8:
    print(f"\n  int8 sha256 (Kaggle wali): {sha256(int8)}")
    print("  (agar ye bf5981da82aad9aa... se shuru hota hai to wahi file hai")
    print("   jo maine receive ki - matlab TRANSFER me NahI, quantization me")
    print("   problem hai; alag hai to transfer bigda tha)")

# ---------------------------------------------------------------------
# battery
# ---------------------------------------------------------------------
BATTERY = [
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
    ("I love this post, very well written", False),
    ("I could kill for a cup of coffee right now", False),
    ("Her presentation killed it, everyone loved it", False),
    ("He was niggardly with his praise", False),
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

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(tokdir)

def encode(texts):
    enc = tok(texts, truncation=True, max_length=128, padding="max_length")
    return (np.array(enc["input_ids"], dtype=np.int64),
            np.array(enc["attention_mask"], dtype=np.int64))

ids, masks = encode([t for t, _ in BATTERY])

import onnxruntime as ort

def battery(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(ids), 8):
        out.extend(float(x) for x in s.run(
            ["score"], {"input_ids": ids[i:i+8], "attention_mask": masks[i:i+8]})[0].ravel())
    return out

def npass(scores):
    return sum(((sc >= 0.5) == exp) for sc, (_, exp) in zip(scores, BATTERY))

def show(tag, scores):
    print(f"\n--- {tag}: {npass(scores)}/24 ---")
    for (t, exp), sc in zip(BATTERY, scores):
        ok = (sc >= 0.5) == exp
        print(f"{'PASS' if ok else 'FAIL'} [{sc:.4f}] {t[:56]}")

print("\n" + "=" * 62)
print("BATTERY - fp32:")
fp_scores = battery(fp32)
show("fp32", fp_scores)

if int8:
    print("\n" + "=" * 62)
    print("BATTERY - int8 (jaisi file ban rahi thi):")
    i8_scores = battery(int8)
    show("int8 (current)", i8_scores)
    md = max(abs(a - b) for a, b in zip(fp_scores, i8_scores))
    print(f"\n  fp32-vs-int8 max diff: {md:.5f}")
    good = md < 0.05 and npass(i8_scores) >= 22
else:
    i8_scores, good = None, False

# ---------------------------------------------------------------------
# agar int8 kharab hai -> naye variants banao
# ---------------------------------------------------------------------
from onnxruntime.quantization import quantize_dynamic, QuantType

best_path, best_scores, best_md = int8, i8_scores, (md if int8 else 9.9)

VARIANTS = [
    ("per_channel QInt8", dict(weight_type=QuantType.QInt8, per_channel=True)),
    ("per_tensor QUInt8", dict(weight_type=QuantType.QUInt8)),
]

if not good and fp32:
    for name, kw in VARIANTS:
        out = f"int8_{name.split()[0]}.onnx"
        try:
            quantize_dynamic(fp32, out, **kw)
        except Exception as e:
            print(f"\n[{name}] quantize FAIL: {e}")
            continue
        sc = battery(out)
        d = max(abs(a - b) for a, b in zip(fp_scores, sc))
        show(f"variant: {name}", sc)
        print(f"  fp32-vs-variant max diff: {d:.5f} | size "
              f"{os.path.getsize(out)/1e6:.1f} MB")
        if d < best_md or (best_path is None):
            best_path, best_scores, best_md = out, sc, d
        if d < 0.05 and npass(sc) >= 22:
            print(f"  --> YE VARIANT THEEK HAI")
            break

# fp16 fallback (agar int8 variants bhi kaam na karein)
# fp32 (268MB) -> fp16 (~134MB) - accuracy lagbhag same rehti hai
if fp32 and best_md >= 0.05:
    print("\n[fp16] int8 variants kaafi acche NahI - fp16 try karta hoon")
    try:
        try:
            from onnxconverter_common import float16
        except ImportError:
            pip("onnxconverter-common")
            from onnxconverter_common import float16
        import onnx as _onnx
        m16 = float16.convert_float_to_float16(_onnx.load(fp32))
        _onnx.save(m16, "distilbert_moderation_v3_fp16.onnx")
        sc = battery("distilbert_moderation_v3_fp16.onnx")
        d = max(abs(a - b) for a, b in zip(fp_scores, sc))
        show("variant: fp16", sc)
        print(f"  fp32-vs-fp16 max diff: {d:.5f} | size "
              f"{os.path.getsize('distilbert_moderation_v3_fp16.onnx')/1e6:.1f} MB")
        if d < best_md:
            best_path = "distilbert_moderation_v3_fp16.onnx"
            best_scores, best_md = sc, d
            print("  --> fp16 winner (abhi ke liye best)")
    except Exception as e:
        print(f"  [fp16] FAIL: {e}")

# ---------------------------------------------------------------------
# deploy zip banao (best int8 + tokenizer)
# ---------------------------------------------------------------------
print("\n" + "=" * 62)
print("DEPLOY ZIP")
if best_path is None or best_scores is None:
    raise SystemExit("Koi bhi theek int8 NahI bana - mujhe output bhejo")

final_name = "distilbert_moderation_v3_int8.onnx"
if os.path.abspath(best_path) != os.path.abspath(final_name):
    if best_path != final_name:
        import shutil
        shutil.copy(best_path, final_name)
print(f"chosen int8: {final_name} ({os.path.getsize(final_name)/1e6:.1f} MB)")
print(f"  battery: {npass(best_scores)}/24 | fp32 se max diff: {best_md:.5f}")
print(f"  sha256: {sha256(final_name)}")

with zipfile.ZipFile("v3-deploy.zip", "w", zipfile.ZIP_STORED) as z:
    z.write(final_name)
    for fn in os.listdir(tokdir):
        z.write(os.path.join(tokdir, fn), f"tokenizer/{fn}")
print(f"\nv3-deploy.zip ready: {os.path.getsize('v3-deploy.zip')/1e6:.1f} MB")
print(f"zip sha256: {sha256('v3-deploy.zip')}")
print("\nYEH ZIP DOWNLOAD KARKE MUJHE BHEJO (attach in chat).")
print("Main iski sha256 se verify karunga - transfer sahi hua ya NahI.")
