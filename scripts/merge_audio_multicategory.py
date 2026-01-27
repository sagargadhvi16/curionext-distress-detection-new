import os
import json
import random
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================
SR = 16000
MIN_SEC = 180          # 3 min
MAX_SEC = 300          # 5 min
WINDOW_SEC = 10        # label granularity
NUM_TEST_SAMPLES = 20
SEED = 42

random.seed(SEED)

PROJECT_ROOT = Path("D:/CN/curionext-distress-detection-new/data/raw/audio")
OUTPUT_DIR = PROJECT_ROOT / "merged_audio_test"
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
        for root, _, fnames in os.walk(p):
            for f in fnames:
                if f.lower().endswith(".wav"):
                    files.append(os.path.join(root, f))
    return files


CATEGORY_FILES = {
    cat: collect_wavs(paths)
    for cat, paths in CATEGORY_PATHS.items()
}

# =========================
# MERGE LOGIC
# =========================
def create_multicategory_sample(idx):
    categories = list(CATEGORY_FILES.keys())
    random.shuffle(categories)

    merged_audio = []
    timeline = []
    total_sec = 0.0

    while total_sec < MIN_SEC:
        cat = random.choice(categories)
        wav_path = random.choice(CATEGORY_FILES[cat])

        audio, _ = librosa.load(wav_path, sr=SR)
        dur = len(audio) / SR

        merged_audio.append(audio)

        num_windows = int(dur // WINDOW_SEC)
        timeline.extend([cat] * max(1, num_windows))

        total_sec += dur

        if total_sec >= MAX_SEC:
            break

    merged_audio = np.concatenate(merged_audio)

    # Trim hard to MAX_SEC
    merged_audio = merged_audio[: MAX_SEC * SR]
    timeline = timeline[: int(len(merged_audio) / (SR * WINDOW_SEC))]

    # Save files
    wav_out = OUTPUT_DIR / f"multicat_test_{idx:02d}.wav"
    json_out = OUTPUT_DIR / f"multicat_test_{idx:02d}.json"

    sf.write(wav_out, merged_audio, SR)

    with open(json_out, "w") as f:
        json.dump(
            {
                "window_sec": WINDOW_SEC,
                "sequence": timeline,
            },
            f,
            indent=2,
        )

    print(f"✔ Saved {wav_out.name} ({len(merged_audio)/SR:.1f}s)")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    for i in range(NUM_TEST_SAMPLES):
        create_multicategory_sample(i)
