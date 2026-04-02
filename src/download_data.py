import os, glob, soundfile as sf, numpy as np, yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANG = "en"
N_SAMPLES = params["n_samples"]
OUT_DIR = f"data/raw/{LANG}/wav"
os.makedirs(OUT_DIR, exist_ok=True)

LIBRI_DIR = "data/raw/LibriSpeech/test-clean"
flac_files = sorted(glob.glob(os.path.join(LIBRI_DIR, "**/*.flac"), recursive=True))
print(f"找到 {len(flac_files)} 个 flac 文件")

def get_transcript(flac_path):
    folder = os.path.dirname(flac_path)
    stem = os.path.splitext(os.path.basename(flac_path))[0]
    for tf in glob.glob(os.path.join(folder, "*.trans.txt")):
        with open(tf) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if parts[0] == stem:
                    return parts[1] if len(parts) > 1 else ""
    return ""

for count, flac_path in enumerate(flac_files[:N_SAMPLES]):
    arr, sr = sf.read(flac_path)
    arr = arr.astype(np.float32)
    wav_path = os.path.join(OUT_DIR, f"commonvoice_{count:06d}.wav")
    sf.write(wav_path, arr, sr)
    print(f"  [{count+1}/{N_SAMPLES}] {wav_path}")

print("完成！")
