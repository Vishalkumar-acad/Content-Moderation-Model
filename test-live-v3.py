"""
LIVE PRODUCTION TEST — pixelabs.in content moderation (v3 DistilBERT)
Ye script LIVE Render API ko test karti hai: https://content-moderation-model.onrender.com
Koi file/model download NahI karna — bas chalao (Colab / Kaggle / laptop, kahin bhi).
"""

import json
import time
import urllib.request

API = "https://content-moderation-model.onrender.com"

# (text, expected_toxic, note)  — 24 cases: 14 benign + 10 toxic
BATTERY = [
    # --- BENIGN (inme se koi bhi toxic NahI hona chahiye) ---
    ("Please put the garbage in the trash", False, ""),
    ("Please take out the garbage", False, "site ka asli case - borderline"),
    ("Take the garbage out to the curb before sunrise", False, ""),
    ("Sorting garbage and recycling is everyone's job", False, ""),
    ("He works as a garbage collector in our city", False, ""),
    ("The garbage truck comes on Monday", False, ""),
    ("Trash pickup in our neighborhood is on Fridays", False, ""),
    ("My phone battery is trash", False, ""),
    ("The trash can is full, someone empty it", False, ""),
    ("Garbage collection is on Tuesdays this week", False, ""),
    ("I love this post, very well written", False, ""),
    ("Her presentation killed it, everyone loved it", False, ""),
    ("He was niggardly with his praise", False, "'niggardly' = kanjoos, gali NahI"),
    ("I could kill for a cup of coffee right now", False, "borderline - thoda garam"),
    # --- TOXIC (ye sab pakde jaane chahiye) ---
    ("You are totally trash", True, ""),
    ("You are trash", True, ""),
    ("You are a worthless piece of garbage", True, ""),
    ("Nobody likes you, just leave", True, "harassment layer pakadta hai"),
    ("Shut up you idiot", True, ""),
    ("Assholes", True, ""),
    ("NIGGER", True, "wordlist pakadta hai"),
    ("You are such a scumbag", True, ""),
    ("Stop being an assclown", True, ""),
    ("That is complete horseshit", True, ""),
]


def call(path, payload=None):
    if payload is None:
        req = urllib.request.Request(API + path)
    else:
        req = urllib.request.Request(
            API + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


# 1) pehle service jagao (Render free tier sota hai, 30-60 sec lag sakta hai)
print("Service jag raha hoon...", flush=True)
version = "?"
for i in range(12):
    try:
        h = call("/")
        version = h.get("version", "?")
        if h.get("status") == "ok":
            break
    except Exception as e:
        print(f"  ({i+1}) abhi so raha hai... {type(e).__name__}")
    time.sleep(10)
print(f"Service LIVE: {version}\n")

# 2) battery chalao
print("=" * 78)
print(f"{'PASS/FAIL':<10} {'SCORE':>7}  {'EXPECT':<7} {'RULE':<18} TEXT")
print("=" * 78)

n_pass = 0
benign_pass = benign_total = toxic_pass = toxic_total = 0
for text, expected, note in BATTERY:
    try:
        r = call("/moderate", {"text": text})
    except Exception as e:
        print(f"ERROR      {type(e).__name__} - service tak NahI pahunch raha: {e}")
        if expected is False:
            benign_total += 1
        else:
            toxic_total += 1
        continue
    got = r["toxic"]
    ok = got == expected
    n_pass += ok
    if expected is False:
        benign_total += 1
        benign_pass += ok
    else:
        toxic_total += 1
        toxic_pass += ok
    rule = r.get("matched_rule", "model")
    mark = "PASS" if ok else "FAIL"
    extra = f"  <- {note}" if note else ""
    print(f"{mark:<10} {r['score']:>7.4f}  {('toxic' if expected else 'clean'):<7} {rule:<18} {text[:40]}{extra}")

print("=" * 78)
print(f"""
RESULT: {n_pass}/24 PASS  ({n_pass/24*100:.0f}%)
  Benign (galat flag NahI):     {benign_pass}/{benign_total}
  Toxic (pakde gaye):           {toxic_pass}/{toxic_total}

NOTE: 'Please take out the garbage' aur 'kill for a cup of coffee'
borderline hain - inka score 0.5-0.8 aana NORMAL hai (flag-only zone,
comment chhupta NahI, sirf review flag lagta hai).
""")
