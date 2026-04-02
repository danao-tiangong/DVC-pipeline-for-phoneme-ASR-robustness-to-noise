import os
import glob
import soundfile as sf
import numpy as np

LANG = "en"
N_SAMPLES = 50
OUT_DIR = f"data/raw/{LANG}/wav"
os.makedirs(OUT_DIR, exist_ok=True)

# LibriSpeech 下载后在这个目录
LIBRI_DIR = "data/raw/LibriSpeech/test-clean"

# 找所有 flac 文件
flac_files = sorted(glob.glob(os.path.join(LIBRI_DIR, "**/*.flac"), recursive=True))
print(f"找到 {len(flac_files)} 个 flac 文件")

# 找对应的文本
def get_transcript(flac_path):
    # trans 文件在同一目录下，格式: {speaker}-{chapter}.trans.txt
    folder = os.path.dirname(flac_path)
    trans_files = glob.glob(os.path.join(folder, "*.trans.txt"))
    stem = os.path.splitext(os.path.basename(flac_path))[0]
    for tf in trans_files:
        with open(tf) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if parts[0] == stem:
                    return parts[1] if len(parts) > 1 else ""
    return ""

count = 0
for flac_path in flac_files[:N_SAMPLES]:
    arr, sr = sf.read(flac_path)
    arr = arr.astype(np.float32)
    wav_path = os.path.join(OUT_DIR, f"commonvoice_{count:06d}.wav")
    sf.write(wav_path, arr, sr)
    transcript = get_transcript(flac_path)
    print(f"  [{count+1}/{N_SAMPLES}] {wav_path} | {transcript[:50]}")
    count += 1

print(f"完成！共处理 {count} 条")
