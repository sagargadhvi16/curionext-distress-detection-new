from fastapi import FastAPI, UploadFile
import librosa

from audio_experiments.xgb_baseline.predict_xgb_asr import predict_xgb_with_asr

app = FastAPI()

@app.post("/audio/predict")
def audio_predict(file: UploadFile):
    # 1. Load audio file into waveform
    audio, sr = librosa.load(file.file, sr=16000, mono=True)

    # 2. Call YOUR pipeline
    result = predict_xgb_with_asr(audio, sr)

    # 3. Return result as-is
    return result
