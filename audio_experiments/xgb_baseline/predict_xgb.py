import joblib
import numpy as np
from audio_experiments.xgb_baseline.features import extract_feature_chunk


MODEL_PATH = "audio_experiments/xgb_baseline/artifacts/your_distress_xgb_3class.joblib"
SCALER_PATH = "audio_experiments/xgb_baseline/artifacts/your_distress_scaler_3class.joblib"
ENCODER_PATH = "audio_experiments/xgb_baseline/artifacts/your_distress_label_encoder_3class.joblib"


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

def predict_xgb(audio_chunk, sr=16000):
    """
    Input: 10s raw audio chunk (numpy array)
    Output: (coarse_label, confidence)
    """
    feat = extract_feature_chunk(audio_chunk, sr)
    feat_scaled = scaler.transform([feat])

    pred_enc = model.predict(feat_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_enc])[0]

    confidence = model.predict_proba(feat_scaled).max()

    return pred_label, confidence
