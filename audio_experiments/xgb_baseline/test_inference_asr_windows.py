import librosa
import numpy as np

from audio_experiments.xgb_baseline.predict_xgb_asr import predict_xgb_with_asr

AUDIO_PATH = "data/CurioNext_Audio/test/multicat_test_01.wav"
SR = 16000
WINDOW_SEC = 10


def chunk_audio(audio, sr, window_sec):
    window_size = int(sr * window_sec)
    windows = []

    for start in range(0, len(audio), window_size):
        chunk = audio[start:start + window_size]

        if len(chunk) == 0:
            continue

        # Pad last window if needed
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))

        windows.append(chunk)

    return windows


# ------------------------------------------------------------------
# Load audio
# ------------------------------------------------------------------
audio, sr = librosa.load(AUDIO_PATH, sr=SR)

windows = chunk_audio(audio, sr, WINDOW_SEC)

print(f"\nTotal windows: {len(windows)}\n")

# ------------------------------------------------------------------
# Window-wise inference
# ------------------------------------------------------------------
for i, window in enumerate(windows):
    out = predict_xgb_with_asr(window)

    print(f"Window {i:02d} [{i*10}-{(i+1)*10}s]")
    print(f"  Label         : {out['audio_label']}")
    print(f"  Confidence    : {out['audio_confidence']:.3f}")
    print(f"  Distress type : {out['distress_type']}")

    if out["asr_text"] is not None:
        print(f"  ASR           : {out['asr_text']}")
    else:
        print(f"  ASR           : <no speech detected>")

    print("-" * 60)
