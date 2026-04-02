import os, json
import editdistance

LANG = "en"
PRED_DIR = f"data/predictions/{LANG}"
SNR_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
METRICS_DIR = "data/metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

def compute_per(ref, pred):
    """把音素序列按空格分割成token列表，计算编辑距离/PER"""
    ref_tokens = ref.strip().split()
    pred_tokens = pred.strip().split()
    if len(ref_tokens) == 0:
        return 0.0
    dist = editdistance.eval(ref_tokens, pred_tokens)
    return dist / len(ref_tokens)

def eval_manifest(pred_path):
    pers = []
    with open(pred_path) as f:
        for line in f:
            rec = json.loads(line)
            per = compute_per(rec["ref_phon"], rec["pred_phon"])
            pers.append(per)
    return sum(pers) / len(pers) if pers else 0.0

results = {}

# clean
clean_pred = os.path.join(PRED_DIR, "clean.jsonl")
per = eval_manifest(clean_pred)
results["clean"] = {"snr_db": None, "per": round(per, 4)}
print(f"clean  -> PER = {per:.4f}")

# noisy
for snr in SNR_LEVELS:
    pred_path = os.path.join(PRED_DIR, f"noisy_snr{snr}.jsonl")
    per = eval_manifest(pred_path)
    results[f"snr_{snr}"] = {"snr_db": snr, "per": round(per, 4)}
    print(f"SNR={snr:2d}dB -> PER = {per:.4f}")

# 保存 metrics
out_path = os.path.join(METRICS_DIR, f"{LANG}_per.json")
with open(out_path, "w") as f:
    json.dump({LANG: results}, f, indent=2)
print(f"\nmetrics 保存到 {out_path}")
