import os, csv, subprocess, yaml
import soundfile as sf
import numpy as np

with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANG = "pa"
N_SAMPLES = params["n_samples"]
SRC_DIR = "data/raw/pa-IN"
OUT_DIR = f"data/raw/{LANG}/wav"
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
with open(os.path.join(SRC_DIR, "validated.tsv"), encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        mp3_path = os.path.join(SRC_DIR, "clips", row["path"])
        if os.path.exists(mp3_path):
            rows.append((mp3_path, row["sentence"]))
        if len(rows) >= N_SAMPLES:
            break

print(f"找到 {len(rows)} 条有效音频")

for i, (mp3_path, sentence) in enumerate(rows):
    wav_path = os.path.join(OUT_DIR, f"commonvoice_{i:06d}.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", mp3_path,
        "-ar", "16000", "-ac", "1", wav_path
    ], capture_output=True)
    print(f"  [{i+1}/{len(rows)}] {wav_path} | {sentence[:40]}")

print("完成！")
