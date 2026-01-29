import os
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

from audio_experiments.xgb_baseline.features import extract_feature_chunk
from audio_experiments.xgb_baseline.labels import fine_to_coarse


SR = 16000
WINDOW_SEC = 10
MAX_FILES_PER_EMO = 10
MAX_WINDOWS_PER_FILE = 10

TRAIN_DIR = "data/CurioNext_Audio/train"

def chunk_audio_fixed(audio, sr, max_windows):
    window_size = sr * WINDOW_SEC
    windows = []

    for i in range(max_windows):
        start = i * window_size
        end = start + window_size
        if start >= len(audio):
            break
        chunk = audio[start:end]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))
        windows.append(chunk)

    return windows

X, y = [], []

for emo in sorted(os.listdir(TRAIN_DIR)):
    emo_dir = os.path.join(TRAIN_DIR, emo)
    if not os.path.isdir(emo_dir):
        continue

    files = [f for f in os.listdir(emo_dir) if f.endswith(".wav")]
    files = files[:MAX_FILES_PER_EMO]

    window_count = 0

    for wav in files:
        path = os.path.join(emo_dir, wav)
        audio, _ = librosa.load(path, sr=SR)

        windows = chunk_audio_fixed(audio, SR, MAX_WINDOWS_PER_FILE)

        for w in windows:
            feat = extract_feature_chunk(w, SR)
            X.append(feat)
            y.append(fine_to_coarse(emo))
            window_count += 1

    print(f"{emo}: {window_count} windows")

X = np.array(X)
y = np.array(y)

print(f"\nTotal training windows: {len(X)}")
print("Classes:", set(y))

label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1
)

model.fit(X_scaled, y_enc)

joblib.dump(model, "your_distress_xgb_3class.joblib")
joblib.dump(scaler, "your_distress_scaler_3class.joblib")
joblib.dump(label_encoder, "your_distress_label_encoder_3class.joblib")

print("\n✅ Training complete. Joblib files saved.")
