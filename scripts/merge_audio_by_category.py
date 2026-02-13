import os
import random
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np

# =========================
# CONFIG
# =========================
SR = 16000
MIN_SEC = 180          # 3 minutes
MAX_SEC = 300          # 5 minutes
TARGET_SAMPLES = 20    # per category
SEED = 42

random.seed(SEED)

PROJECT_ROOT = Path("D:/CN/curionext-distress-detection-new/data/raw/audio")
OUTPUT_DIR = PROJECT_ROOT / "merged_individual_category_audio"
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# CATEGORY → DATA SOURCES
# =========================
CATEGORY_PATHS = {
    "cry": [
        PROJECT_ROOT / "test_cry",
        PROJECT_ROOT / "train_cry",
    ],

    "fear": [
        PROJECT_ROOT / "besd/ENGLISH/FEAR",
    ],

    "anger": [
        PROJECT_ROOT / "BESD/ENGLISH/anger",
    ],

    "abuse": [
        PROJECT_ROOT / "prima/SC_audio_Hindi",
    ],

    "scream": [
        PROJECT_ROOT / "screaming/Screaming",
    ],

    "neutral": [
        PROJECT_ROOT / "BESD/neutral",
        PROJECT_ROOT / "screaming/NotScreaming",
        PROJECT_ROOT / "esc50",
    ],
}

# =========================
# UTILS
# =========================
def collect_wavs(paths):
    files = []
    for p in paths:
        for root, _, filenames in os.walk(p):
            for f in filenames:
                if f.lower().endswith(".wav"):
                    files.append(os.path.join(root, f))
    return files


def merge_category(category, source_paths):
    wavs = collect_wavs(source_paths)
    print(f"\n[INFO] {category}: {len(wavs)} source files")

    if len(wavs) == 0:
        print(f"[WARN] No files found for {category}, skipping.")
        return

    cat_out_dir = OUTPUT_DIR / category
    cat_out_dir.mkdir(exist_ok=True)

    sample_idx = 0

    while sample_idx < TARGET_SAMPLES:
        random.shuffle(wavs)

        merged = []
        total_sec = 0.0

        for wav_path in wavs:
            audio, _ = librosa.load(wav_path, sr=SR)
            audio_sec = len(audio) / SR

            # stop if adding this clip exceeds MAX_SEC
            if total_sec + audio_sec > MAX_SEC:
                break

            merged.append(audio)
            total_sec += audio_sec

            if total_sec >= MIN_SEC:
                break

        # safety check
        if total_sec < MIN_SEC:
            print(f"[WARN] Not enough audio to build sample for {category}")
            break

        merged_audio = np.concatenate(merged)

        out_path = cat_out_dir / f"{category}_{sample_idx:02d}.wav"
        sf.write(out_path, merged_audio, SR)

        print(f"  ✔ saved {out_path.name} ({total_sec:.1f}s)")
        sample_idx += 1


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    for category, paths in CATEGORY_PATHS.items():
        merge_category(category, paths)
