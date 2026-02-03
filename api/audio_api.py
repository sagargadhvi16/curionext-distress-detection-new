from fastapi import FastAPI, UploadFile, File
import librosa

from audio_experiments.xgb_baseline.predict_xgb_asr import predict_xgb_with_asr

app = FastAPI(
    title="Audio Distress Inference API",
    version="1.0.0"
)

@app.post("/audio/predict")
def audio_predict(file: UploadFile = File(...)):
    """
    Takes an audio file and returns:
    - audio_label
    - confidence
    - distress_type
    - asr_text (if available)
    """

    # Load audio
    audio, sr = librosa.load(file.file, sr=16000, mono=True)

    # Run YOUR model pipeline
    result = predict_xgb_with_asr(audio, sr)

    # Must return JSON-serializable dict
    return {
        "audio_label": result.get("audio_label"),
        "audio_confidence": float(result.get("audio_confidence", 0.0)),
        "distress_type": result.get("distress_type"),
        "asr_text": result.get("asr_text"),
    }
