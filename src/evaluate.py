import os, json, re
import editdistance

LANG = "en"
PRED_DIR = f"data/predictions/{LANG}"
SNR_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
METRICS_DIR = "data/metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

# IPA 重音符号和声调符号，计算时去掉
STRESS_MARKS = set("ˈˌ")

def normalize(phon_str):
    """
    把音素字符串统一成单个音素的列表，去掉重音符号。
    不管是 'hiː hˈəʊpt' 还是 'h iː h oʊ p t'，
    都先去掉重音符号，再按空格分割成字符级 token。
    """
    # 去掉重音符号
    cleaned = "".join(c for c in phon_str if c not in STRESS_MARKS)
    # 按空格分割（pred 已经是空格分隔；ref 按词分隔后再拆字符）
    tokens = []
    for word in cleaned.split():
        # 把每个"词"拆成单个字符（处理 ref 的合并音素）
        # 但要保留多字符音素如 oʊ, iː, ɔː 等
        # 方法：直接把字符串拆成 unicode 字符序列
        for ch in word:
            tokens.append(ch)
    return tokens

def compute_per(ref, pred):
    ref_tokens = normalize(ref)
    pred_tokens = normalize(pred)
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
