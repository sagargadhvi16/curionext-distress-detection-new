import os
import json
import librosa
import numpy as np
from collections import Counter

from audio_experiments.xgb_baseline.labels import fine_to_coarse
from audio_experiments.xgb_baseline.predict_xgb import predict_xgb


SR = 16000
WINDOW_SEC = 10
TEST_DIR = "data/CurioNext_Audio/test"


def chunk_audio(audio, sr, window_sec=10):
    window_size = sr * window_sec
    chunks = []
    for start in range(0, len(audio), window_size):
        end = start + window_size
        chunk = audio[start:end]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))
        chunks.append(chunk)
    return chunks


total_correct = 0
total_windows = 0

seq_correct = 0
seq_total = 0

for file in sorted(os.listdir(TEST_DIR)):
    if not file.endswith(".wav"):
        continue

    base = file.replace(".wav", "")
    wav_path = os.path.join(TEST_DIR, file)
    json_path = os.path.join(TEST_DIR, base + ".json")

    print(f"\n=== Testing {base} ===")

    audio, _ = librosa.load(wav_path, sr=SR)
    windows = chunk_audio(audio, SR, WINDOW_SEC)

    try:
        with open(json_path, "r") as f:
            gt_data = json.load(f)
        gt_seq = [fine_to_coarse(lbl) for lbl in gt_data["sequence"]]
    except Exception as e:
        print(f"❌ Missing or invalid JSON for {base}: {e}")
        continue

    pred_seq = []
    for w in windows[:len(gt_seq)]:
        label, conf = predict_xgb(w)
        pred_seq.append(label)

    L = min(len(pred_seq), len(gt_seq))
    pred_seq = pred_seq[:L]
    gt_seq = gt_seq[:L]

    # ---- Window-level accuracy ----
    correct = sum(p == g for p, g in zip(pred_seq, gt_seq))
    acc = correct / L if L > 0 else 0

    total_correct += correct
    total_windows += L

    print("Pred (coarse):", pred_seq)
    print("GT   (coarse):", gt_seq)
    print(f"Window accuracy: {correct}/{L} = {acc:.2f}")

    # ---- Sequence-level accuracy (majority vote) ----
    pred_major = Counter(pred_seq).most_common(1)[0][0]
    gt_major = Counter(gt_seq).most_common(1)[0][0]

    seq_match = pred_major == gt_major
    seq_correct += int(seq_match)
    seq_total += 1

    print(f"Seq Pred: {pred_major} | Seq GT: {gt_major} | Match: {seq_match}")


# ---- Overall metrics ----
overall_window_acc = total_correct / total_windows if total_windows > 0 else 0
overall_seq_acc = seq_correct / seq_total if seq_total > 0 else 0

print("\n=== OVERALL RESULTS ===")
print(f"Window-level accuracy: {total_correct}/{total_windows} = {overall_window_acc:.2f}")
print(f"Sequence-level accuracy: {seq_correct}/{seq_total} = {overall_seq_acc:.2f}")

''' 
#Window-level accuracy (0.49)

- Audio is split into 10-second windows

- The model predicts one coarse emotion per window

- We check how many windows are predicted correctly

- 👉 About 49% of individual 10-second segments are classified correctly

- This shows how well the model works on short, fine-grained audio chunks, which is hard because emotions change quickly.

# Sequence-level accuracy (0.90)

- Each audio file has many windows

- We take the most frequent predicted emotion across all windows

- We compare it with the dominant ground-truth emotion of the file

- 👉 90% of full audio files have the correct dominant emotion predicted
'''