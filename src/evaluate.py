import os, json
import editdistance
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANG = params["lang"]
SNR_LEVELS = params["snr_levels"]
PRED_DIR = f"data/predictions/{LANG}"
METRICS_DIR = "data/metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

STRESS_MARKS = set("ˈˌ")

def normalize(phon_str):
    cleaned = "".join(c for c in phon_str if c not in STRESS_MARKS)
    tokens = []
    for word in cleaned.split():
        for ch in word:
            tokens.append(ch)
    return tokens

def compute_per(ref, pred):
    ref_tokens = normalize(ref)
    pred_tokens = normalize(pred)
    if len(ref_tokens) == 0:
        return 0.0
    return editdistance.eval(ref_tokens, pred_tokens) / len(ref_tokens)

def eval_manifest(pred_path):
    pers = []
    with open(pred_path) as f:
        for line in f:
            rec = json.loads(line)
            pers.append(compute_per(rec["ref_phon"], rec["pred_phon"]))
    return sum(pers) / len(pers) if pers else 0.0

results = {}
per = eval_manifest(f"{PRED_DIR}/clean.jsonl")
results["clean"] = {"snr_db": None, "per": round(per, 4)}
print(f"[evaluate] clean -> PER={per:.4f}")

for snr in SNR_LEVELS:
    per = eval_manifest(f"{PRED_DIR}/noisy_snr{snr}.jsonl")
    results[f"snr_{snr}"] = {"snr_db": snr, "per": round(per, 4)}
    print(f"[evaluate] SNR={snr:2d}dB -> PER={per:.4f}")

out_path = f"{METRICS_DIR}/{LANG}_per.json"
with open(out_path, "w") as f:
    json.dump({LANG: results}, f, indent=2)
print(f"[evaluate] metrics -> {out_path}")
