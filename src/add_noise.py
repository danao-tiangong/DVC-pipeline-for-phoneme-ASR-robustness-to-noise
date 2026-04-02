import os, json, shutil, hashlib, argparse
import numpy as np
import soundfile as sf
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--lang", required=True)
args = parser.parse_args()

with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANG = args.lang
SNR_LEVELS = params["snr_levels"]
SEED = params["noise_seed"]
CLEAN_MANIFEST = f"data/manifests/{LANG}/clean.jsonl"

def add_noise(signal, snr_db, rng):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal.shape)
    return signal + noise

def add_noise_to_file(input_wav, output_wav, snr_db, seed=None):
    signal, sr = sf.read(input_wav)
    if signal.ndim != 1:
        raise ValueError("Only mono audio is supported")
    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)
    sf.write(output_wav, noisy_signal, sr)

def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    with open(CLEAN_MANIFEST) as f:
        records = [json.loads(line) for line in f]
    print(f"[add_noise] lang={LANG}, SNR levels={SNR_LEVELS}")
    for snr in SNR_LEVELS:
        out_wav_dir = f"data/raw/{LANG}/noisy/snr_{snr}"
        out_manifest = f"data/manifests/{LANG}/noisy_snr{snr}.jsonl"
        os.makedirs(out_wav_dir, exist_ok=True)
        tmp_manifest = out_manifest + ".tmp"
        with open(tmp_manifest, "w") as f:
            for rec in records:
                stem = os.path.basename(rec["wav_path"])
                noisy_wav = os.path.join(out_wav_dir, stem)
                add_noise_to_file(rec["wav_path"], noisy_wav, snr_db=snr, seed=SEED)
                new_rec = dict(rec)
                new_rec["wav_path"] = noisy_wav
                new_rec["snr_db"] = snr
                new_rec["audio_md5"] = file_md5(noisy_wav)
                f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
        shutil.move(tmp_manifest, out_manifest)
        print(f"  SNR={snr}dB -> {out_manifest}")
    print("[add_noise] done")

if __name__ == "__main__":
    main()
