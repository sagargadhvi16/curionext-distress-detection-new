"""
Synthetic Audio Generation for Child Distress Detection (FINAL – AGE 3+)

Kept sources:
- crying_generic                     → real cry
- crying_structured/{discomfort,tired} → auxiliary distress
- screaming/Screaming                → high-arousal distress
- RAVDESS (anger, fear, sad)          → adult speech → child distress
- BESD (ANGER, FEAR, SAD only)        → bilingual robustness
- ESC-50                              → background noise only

Removed:
- hungry, burping, belly_pain
- BESD: disgust, happy, neutral

Goal:
Acoustic similarity for distress detection (not realism, not voice cloning)

Aligned with CurioNext MVP.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import random
import csv

# -----------------------------
# Core parameters
# -----------------------------
SR = 16000
DURATION_RANGE = (3.0, 6.0)
TARGET_SAMPLES = 500

OUT_DIR = Path("data/synthetic/audio/distress")
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = OUT_DIR / "metadata.csv"

# -----------------------------
# Dataset paths
# -----------------------------
BASE = Path("data/raw/audio")

CRY_GENERIC = BASE / "crying_generic"
CRY_STRUCT = BASE / "crying_structured"
SCREAMING = BASE / "screaming" / "Screaming"
RAVDESS = BASE / "ravdess"
BESD = BASE / "besd"
ESC50 = BASE / "esc50"

# -----------------------------
# Utility
# -----------------------------
def load_audio(path: Path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)

def normalize(audio):
    return audio / (np.max(np.abs(audio)) + 1e-6)

# -----------------------------
# Augmentations (child-like)
# -----------------------------
def pitch_shift(audio):
    return librosa.effects.pitch_shift(
        audio, sr=SR, n_steps=random.uniform(3, 7)
    )

def time_stretch(audio):
    return librosa.effects.time_stretch(
        audio, rate=random.uniform(0.85, 1.15)
    )

def emphasize_high_freq(audio):
    return librosa.effects.preemphasis(audio, coef=0.97)

def cry_burst_envelope(audio):
    mask = np.zeros_like(audio)
    pos = 0
    while pos < len(audio):
        burst = int(random.uniform(0.2, 0.6) * SR)
        pause = int(random.uniform(0.1, 0.4) * SR)
        mask[pos:pos + burst] = 1
        pos += burst + pause
    return audio * mask

def add_background_noise(audio):
    if not ESC50.exists() or random.random() > 0.6:
        return audio, "none"

    noise = load_audio(random.choice(list(ESC50.glob("*.wav"))))
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio)/len(noise))))
    noise = noise[:len(audio)]

    snr = random.uniform(5, 15)
    scale = np.sqrt(np.mean(audio**2) / (10**(snr/10) * np.mean(noise**2) + 1e-9))
    return audio + scale * noise, "esc50"

# -----------------------------
# Dataset collectors
# -----------------------------
def collect_wavs(folder):
    return list(folder.rglob("*.wav")) if folder.exists() else []

def collect_cry_structured():
    files = []
    for sub in ["discomfort", "tired"]:
        files += collect_wavs(CRY_STRUCT / sub)
    return files

def collect_besd():
    files = []
    for lang in ["ENGLISH", "TELUGU"]:
        for emo in ["ANGER", "FEAR", "SAD"]:
            files += collect_wavs(BESD / lang / emo)
    return files

def collect_ravdess():
    files = []
    for wav in RAVDESS.rglob("*.wav"):
        try:
            emotion = int(wav.stem.split("-")[2])
            if emotion in {4, 5, 6}:  # sad, angry, fearful
                files.append(wav)
        except:
            pass
    return files

# -----------------------------
# Source pools (balanced)
# -----------------------------
SOURCES = [
    ("crying_generic", collect_wavs(CRY_GENERIC), 0.35),
    ("crying_structured", collect_cry_structured(), 0.15),
    ("screaming", collect_wavs(SCREAMING), 0.20),
    ("ravdess", collect_ravdess(), 0.20),
    ("besd", collect_besd(), 0.10),
]

# -----------------------------
# Generation
# -----------------------------
def generate_one(idx, writer):
    r, acc = random.random(), 0
    for name, pool, p in SOURCES:
        acc += p
        if r <= acc and pool:
            src = random.choice(pool)
            src_name = name
            break
    else:
        return

    audio = load_audio(src)
    target_len = int(random.uniform(*DURATION_RANGE) * SR)
    #audio = audio[:target_len]

#4-5 sec syn audio
    if len(audio) >= target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))


    audio = emphasize_high_freq(audio)
    audio = pitch_shift(audio)
    audio = time_stretch(audio)
    audio = cry_burst_envelope(audio)
    audio, noise = add_background_noise(audio)
    audio = normalize(audio)

    out = f"distress_{idx:05d}.wav"
    sf.write(OUT_DIR / out, audio, SR)

    writer.writerow({
        "file": out,
        "source": src_name,
        "original_path": str(src),
        "noise": noise
    })

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    print("Generating synthetic distress audio (age 3+)")

    with open(META_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "source", "original_path", "noise"]
        )
        writer.writeheader()

        for i in range(TARGET_SAMPLES):
            generate_one(i, writer)
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{TARGET_SAMPLES}")

    print("✅ Done")
