import os, json, shutil
import torch
import soundfile as sf
import numpy as np
import yaml
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

with open("params.yaml") as f:
    params = yaml.safe_load(f)

LANG = params["lang"]
MODEL_ID = params["model_id"]
TARGET_SR = params["target_sr"]
SNR_LEVELS = params["snr_levels"]
MANIFEST_DIR = f"data/manifests/{LANG}"
PRED_DIR = f"data/predictions/{LANG}"
os.makedirs(PRED_DIR, exist_ok=True)

print(f"[run_inference] loading model {MODEL_ID} ...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()
print("[run_inference] model loaded")

def predict_manifest(manifest_path, out_path):
    with open(manifest_path) as f:
        records = [json.loads(line) for line in f]
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as fout:
        for i, rec in enumerate(records):
            signal, sr = sf.read(rec["wav_path"])
            signal = signal.astype(np.float32)
            if sr != TARGET_SR:
                raise ValueError(f"Expected {TARGET_SR}Hz, got {sr}Hz")
            inputs = processor(signal, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_phon = processor.batch_decode(pred_ids)[0]
            out_rec = dict(rec)
            out_rec["pred_phon"] = pred_phon
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            if (i+1) % 10 == 0 or i == 0:
                print(f"    [{i+1}/{len(records)}] {rec['utt_id']}")
    shutil.move(tmp_path, out_path)

print(f"\n[run_inference] lang={LANG}")
predict_manifest(f"{MANIFEST_DIR}/clean.jsonl", f"{PRED_DIR}/clean.jsonl")
for snr in SNR_LEVELS:
    print(f"  SNR={snr}dB...")
    predict_manifest(f"{MANIFEST_DIR}/noisy_snr{snr}.jsonl", f"{PRED_DIR}/noisy_snr{snr}.jsonl")

print("[run_inference] done")
