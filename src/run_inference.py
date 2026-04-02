import os, json, shutil, glob
import torch
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
LANG = "en"
MANIFEST_DIR = f"data/manifests/{LANG}"
PRED_DIR = f"data/predictions/{LANG}"
SNR_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
TARGET_SR = 16000

os.makedirs(PRED_DIR, exist_ok=True)

print(f"加载模型 {MODEL_ID} ...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()
print("模型加载完成！")

def predict_manifest(manifest_path, out_path):
    with open(manifest_path) as f:
        records = [json.loads(line) for line in f]

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as fout:
        for i, rec in enumerate(records):
            wav_path = rec["wav_path"]
            signal, sr = sf.read(wav_path)
            signal = signal.astype(np.float32)

            # 确保是 16kHz
            if sr != TARGET_SR:
                raise ValueError(f"期望16kHz，实际{sr}Hz: {wav_path}")

            inputs = processor(signal, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            pred_phon = processor.batch_decode(predicted_ids)[0]

            out_rec = dict(rec)
            out_rec["pred_phon"] = pred_phon
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            if (i+1) % 10 == 0 or i == 0:
                print(f"    [{i+1}/{len(records)}] ref: {rec['ref_phon'][:30]} | pred: {pred_phon[:30]}")

    shutil.move(tmp_path, out_path)

# 先处理 clean
clean_manifest = os.path.join(MANIFEST_DIR, "clean.jsonl")
clean_pred = os.path.join(PRED_DIR, "clean.jsonl")
print(f"\n处理 clean...")
predict_manifest(clean_manifest, clean_pred)

# 处理各个 SNR 等级
for snr in SNR_LEVELS:
    manifest = os.path.join(MANIFEST_DIR, f"noisy_snr{snr}.jsonl")
    pred_out = os.path.join(PRED_DIR, f"noisy_snr{snr}.jsonl")
    print(f"\n处理 SNR={snr}dB ...")
    predict_manifest(manifest, pred_out)

print("\n全部推理完成！")
