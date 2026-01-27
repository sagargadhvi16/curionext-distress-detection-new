from pathlib import Path
import librosa
import soundfile as sf

# ---------------- CONFIG ----------------
CHUNK_SEC = 5          # use 5 or 10
SR = 16000
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "chunks"
OUTPUT_DIR.mkdir(exist_ok=True)

def chunk_wav(wav_path: Path):
    audio, sr = librosa.load(wav_path, sr=SR)
    chunk_len = CHUNK_SEC * SR
    total_len = len(audio)

    out_dir = OUTPUT_DIR / wav_path.parent.name / wav_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_idx = 0
    for start in range(0, total_len, chunk_len):
        end = start + chunk_len
        if end > total_len:
            break  # drop last partial chunk

        chunk = audio[start:end]
        out_path = out_dir / f"chunk_{chunk_idx:03d}.wav"
        sf.write(out_path, chunk, SR)
        chunk_idx += 1

    print(f"Chunked {wav_path.name} → {chunk_idx} chunks")

# -------- RUN FOR ALL MEDIUM & LONG FILES --------
for wav in INPUT_DIR.rglob("*.wav"):
    if "short" in wav.stem.lower():
        continue  # skip short clips
    chunk_wav(wav)
