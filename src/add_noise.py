import os, json, shutil, glob
import numpy as np
import soundfile as sf

SNR_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
LANG = "en"
CLEAN_MANIFEST = f"data/manifests/{LANG}/clean.jsonl"
SEED = 42

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

def main():
    # 读取 clean manifest
    with open(CLEAN_MANIFEST) as f:
        records = [json.loads(line) for line in f]

    for snr in SNR_LEVELS:
        out_wav_dir = f"data/raw/{LANG}/noisy/snr_{snr}"
        out_manifest_dir = f"data/manifests/{LANG}"
        out_manifest = os.path.join(out_manifest_dir, f"noisy_snr{snr}.jsonl")
        os.makedirs(out_wav_dir, exist_ok=True)
        os.makedirs(out_manifest_dir, exist_ok=True)

        print(f"\nSNR = {snr} dB")
        tmp_manifest = out_manifest + ".tmp"

        with open(tmp_manifest, "w") as f:
            for rec in records:
                stem = os.path.basename(rec["wav_path"])
                noisy_wav = os.path.join(out_wav_dir, stem)

                add_noise_to_file(rec["wav_path"], noisy_wav, snr_db=snr, seed=SEED)

                new_rec = dict(rec)
                new_rec["wav_path"] = noisy_wav
                new_rec["snr_db"] = snr
                f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                print(f"  {rec['utt_id']} -> {noisy_wav}")

        shutil.move(tmp_manifest, out_manifest)
        print(f"  manifest -> {out_manifest}")

    print("\n完成！所有噪声等级处理完毕")

if __name__ == "__main__":
    main()
