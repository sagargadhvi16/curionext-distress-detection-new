from fastapi import FastAPI, UploadFile
import librosa
import requests

from audio_experiments.xgb_baseline.predict_xgb_asr import predict_xgb_with_asr

app = FastAPI()

RETRIEVAL_URL = "http://127.0.0.1:8001/retrieve/similar"

@app.post("/audio/analyze")
def analyze_audio(file: UploadFile):
    # 1. Load audio
    audio, sr = librosa.load(file.file, sr=16000, mono=True)

    # 2. Audio inference
    audio_result = predict_xgb_with_asr(audio, sr)

    # 3. MAP → retrieval payload
    retrieval_payload = {
        "emotion_state": audio_result["audio_label"],
        "severity": int(audio_result["audio_confidence"] * 10),
        "asr_text": audio_result["asr_text"]
    }

    # 4. Call retrieval service
    retrieval_response = requests.post(
        RETRIEVAL_URL,
        json=retrieval_payload,
        timeout=10
    )

    retrieval_data = retrieval_response.json()

    # 5. Return combined output
    return {
        "audio_result": audio_result,
        "retrieval_result": retrieval_data
    }
