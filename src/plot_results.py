import json
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

METRICS_FILE = "data/metrics/en_per.json"
PLOTS_DIR = "data/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

with open(METRICS_FILE) as f:
    all_metrics = json.load(f)

fig, ax = plt.subplots(figsize=(9, 5))

for lang, results in all_metrics.items():
    snr_vals = []
    per_vals = []
    clean_per = None

    for key, val in results.items():
        if val["snr_db"] is None:
            clean_per = val["per"]
        else:
            snr_vals.append(val["snr_db"])
            per_vals.append(val["per"])

    # 按 SNR 排序
    pairs = sorted(zip(snr_vals, per_vals))
    snr_vals = [p[0] for p in pairs]
    per_vals = [p[1] for p in pairs]

    ax.plot(snr_vals, per_vals, marker="o", linewidth=2, label=f"{lang}")
    if clean_per is not None:
        ax.axhline(y=clean_per, linestyle="--", alpha=0.5,
                   label=f"{lang} clean (PER={clean_per:.3f})")

ax.set_xlabel("SNR (dB)", fontsize=12)
ax.set_ylabel("PER (Phoneme Error Rate)", fontsize=12)
ax.set_title("Phoneme ASR Robustness to Noise", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.invert_xaxis()  # 左边噪音大，右边噪音小，更直观

out_path = os.path.join(PLOTS_DIR, "per_vs_snr.png")
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"图保存到 {out_path}")
