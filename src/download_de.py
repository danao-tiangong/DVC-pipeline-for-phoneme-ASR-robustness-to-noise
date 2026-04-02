import os
import glob
import soundfile as sf
import numpy as np

LANG = "de"
N_SAMPLES = 50
OUT_DIR = f"data/raw/{LANG}/wav"
os.makedirs(OUT_DIR, exist_ok=True)

import urllib.request, tarfile

URL = "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"
TAR_PATH = "data/raw/mls_de.tar.gz"

if not os.path.exists("data/raw/MLS"):
    print("downloading MLS German dev set...")
    urllib.request.urlretrieve(URL, TAR_PATH)
    with tarfile.open(TAR_PATH) as tar:
        tar.extractall("data/raw/")
    print("extracted!")

flac_files = sorted(glob.glob("data/raw/MLS/**/*.flac", recursive=True))
print(f"found {len(flac_files)} flac files")

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

print("done!")
