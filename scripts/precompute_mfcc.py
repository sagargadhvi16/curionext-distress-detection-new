"""
Precompute MFCC features for audio distress detection.
Outputs:
- data/processed/audio/features/*.npy
- data/processed/audio/labels.csv
"""

import csv
from pathlib import Path
import numpy as np

from src.audio.preprocessing import load_audio, normalize_audio
from src.audio.features import extract_mfcc


# -------------------------------------------------
# Paths
# -------------------------------------------------
RAW_ROOT = Path("data/raw/audio")
OUT_ROOT = Path("data/processed/audio")
FEAT_DIR = OUT_ROOT / "features"
LABEL_FILE = OUT_ROOT / "labels.csv"

FEAT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Helper
# -------------------------------------------------
def save_feature(idx, mfcc):
    out_path = FEAT_DIR / f"{idx:06d}.npy"
    np.save(out_path, mfcc)
    return out_path.name


# -------------------------------------------------
# Dataset collectors
# -------------------------------------------------
def collect_wavs(folder):
    return list(folder.rglob("*.wav")) if folder.exists() else []


def collect_ravdess():
    files = []
    for wav in (RAW_ROOT / "ravdess").rglob("*.wav"):
        try:
            emotion = int(wav.stem.split("-")[2])
            if emotion in {4, 5, 6}:  # sad, angry, fear
                files.append(wav)
        except Exception:
            pass
    return files


def collect_besd():
    files = []
    besd = RAW_ROOT / "besd"

    for lang in ["ENGLISH", "HINDI", "TELUGU"]:
        for emo in ["ANGER", "FEAR", "SAD"]:
            files += collect_wavs(besd / lang / emo)

    return files


def collect_besd_non_distress():
    files = []
    besd = RAW_ROOT / "besd"

    for lang in ["ENGLISH", "HINDI", "TELUGU"]:
        for emo in ["HAPPY", "NEUTRAL"]:
            files += collect_wavs(besd / lang / emo)

    return files


# -------------------------------------------------
# Source pools
# -------------------------------------------------
DISTRESS_SOURCES = [
    ("train_cry", collect_wavs(RAW_ROOT / "train_cry")),
    ("test_cry", collect_wavs(RAW_ROOT / "test_cry")),
    ("screaming", collect_wavs(RAW_ROOT / "screaming" / "Screaming")),
    ("ravdess", collect_ravdess()),
    ("besd", collect_besd()),
    ("adema", collect_wavs(RAW_ROOT / "adema")),
]

NON_DISTRESS_SOURCES = [
    ("esc50", collect_wavs(RAW_ROOT / "esc50")),
    ("not_screaming", collect_wavs(RAW_ROOT / "screaming" / "NotScreaming")),
    ("besd_non", collect_besd_non_distress()),
]


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    idx = 0
    rows = []

    print("🚀 Starting MFCC precomputation")

    for label, sources in [(1, DISTRESS_SOURCES), (0, NON_DISTRESS_SOURCES)]:
        for name, files in sources:
            print(f"Processing {name} | label={label} | files={len(files)}")

            for wav in files:
                try:
                    audio, sr = load_audio(wav)
                    audio = normalize_audio(audio)

                    mfcc = extract_mfcc(audio, sr)  # (39, T)

                    feat_name = save_feature(idx, mfcc)

                    rows.append({
                        "id": feat_name,
                        "label": label,
                        "source": name,
                        "path": str(wav)
                    })

                    idx += 1

                except Exception as e:
                    print(f"❌ Skipped {wav}: {e}")

    # Save labels
    with open(LABEL_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "label", "source", "path"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Done! Total samples: {idx}")
    print(f"📁 Features saved to: {FEAT_DIR}")
    print(f"📄 Labels saved to: {LABEL_FILE}")


if __name__ == "__main__":
    main()
