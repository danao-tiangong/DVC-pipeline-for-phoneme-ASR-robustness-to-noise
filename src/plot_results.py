import json, os
import matplotlib.pyplot as plt
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANGS = params["langs"]
METRICS_DIR = "data/metrics"
PLOTS_DIR = "data/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# 合并所有语言的 metrics
all_metrics = {}
for lang in LANGS:
    metrics_file = f"{METRICS_DIR}/{lang}_per.json"
    with open(metrics_file) as f:
        all_metrics.update(json.load(f))

fig, ax = plt.subplots(figsize=(9, 5))
mean_per_by_snr = {}

for lang, results in all_metrics.items():
    snr_vals, per_vals, clean_per = [], [], None
    for key, val in results.items():
        if val["snr_db"] is None:
            clean_per = val["per"]
        else:
            snr_vals.append(val["snr_db"])
            per_vals.append(val["per"])
            mean_per_by_snr.setdefault(val["snr_db"], []).append(val["per"])

    pairs = sorted(zip(snr_vals, per_vals))
    snr_vals = [p[0] for p in pairs]
    per_vals = [p[1] for p in pairs]
    ax.plot(snr_vals, per_vals, marker="o", linewidth=2, label=f"{lang}")
    if clean_per is not None:
        ax.axhline(y=clean_per, linestyle="--", alpha=0.4, label=f"{lang} clean ({clean_per:.3f})")

# 跨语言均值曲线（多语言时才画）
if len(LANGS) > 1:
    mean_snrs = sorted(mean_per_by_snr.keys())
    mean_pers = [sum(mean_per_by_snr[s]) / len(mean_per_by_snr[s]) for s in mean_snrs]
    ax.plot(mean_snrs, mean_pers, marker="s", linewidth=2.5,
            linestyle="--", color="black", label="mean")

ax.set_xlabel("SNR (dB)", fontsize=12)
ax.set_ylabel("PER (Phoneme Error Rate)", fontsize=12)
ax.set_title("Phoneme ASR Robustness to Noise", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

out_path = f"{PLOTS_DIR}/per_vs_snr.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"[plot] saved -> {out_path}")
