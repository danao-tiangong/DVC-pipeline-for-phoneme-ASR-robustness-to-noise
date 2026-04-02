import os, json, hashlib, subprocess, glob, shutil
import soundfile as sf
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANG = params["lang"]
WAV_DIR = f"data/raw/{LANG}/wav"
OUT_DIR = f"data/manifests/{LANG}"
OUT_FILE = os.path.join(OUT_DIR, "clean.jsonl")
LIBRI_DIR = f"data/raw/LibriSpeech/test-clean"

def load_transcripts():
    transcripts = {}
    for tf in glob.glob(os.path.join(LIBRI_DIR, "**/*.trans.txt"), recursive=True):
        with open(tf) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]
    return transcripts

def text_to_phonemes(text, lang="en"):
    result = subprocess.run(
        ["espeak-ng", "--ipa", "-q", "-v", lang, text],
        capture_output=True, text=True
    )
    return result.stdout.strip().replace("\n", " ")

def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    transcripts = load_transcripts()
    libri_keys = sorted(transcripts.keys())
    wav_files = sorted(glob.glob(os.path.join(WAV_DIR, "*.wav")))
    print(f"[make_manifest] lang={LANG}, {len(wav_files)} fichiers wav")

    tmp_file = OUT_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        for i, wav_path in enumerate(wav_files):
            stem = os.path.splitext(os.path.basename(wav_path))[0]
            utt_id = f"{LANG}_{stem}"
            info = sf.info(wav_path)
            ref_text = transcripts[libri_keys[i]].upper() if i < len(libri_keys) else ""
            ref_phon = text_to_phonemes(ref_text)
            audio_md5 = file_md5(wav_path)
            record = {
                "utt_id": utt_id, "lang": LANG, "wav_path": wav_path,
                "ref_text": ref_text, "ref_phon": ref_phon,
                "sr": info.samplerate, "duration_s": round(info.duration, 3),
                "audio_md5": audio_md5, "snr_db": None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  [{i+1}/{len(wav_files)}] {utt_id}")

    shutil.move(tmp_file, OUT_FILE)
    print(f"[make_manifest] done -> {OUT_FILE}")

if __name__ == "__main__":
    main()
