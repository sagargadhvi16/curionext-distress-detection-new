from pathlib import Path
from pydub import AudioSegment
import random

EMOTION = "cry"
TARGET_MIN = 3
SILENCE_MS = 700
SR = 16000

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input" / EMOTION
OUTPUT_DIR = BASE_DIR / "output" / EMOTION / f"{TARGET_MIN}min"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_MS = TARGET_MIN * 60 * 1000
silence = AudioSegment.silent(duration=SILENCE_MS)

files = list(INPUT_DIR.glob("*"))
random.shuffle(files)

merged = AudioSegment.silent(duration=0)
valid_files = 0

for f in files:
    try:
        audio = AudioSegment.from_file(f)
        audio = audio.set_frame_rate(SR).set_channels(1).set_sample_width(2)

        if len(audio) == 0:
            print(f"⚠️ Empty audio skipped: {f.name}")
            continue

        merged += audio + silence
        valid_files += 1

        if len(merged) >= TARGET_MS:
            break

    except Exception as e:
        print(f"❌ Failed to load {f.name}: {e}")

print("Valid clips merged:", valid_files)
print("Final duration (sec):", len(merged) / 1000)

if len(merged) == 0:
    raise RuntimeError("Merged audio is EMPTY. Check input files.")

out_file = OUTPUT_DIR / f"{EMOTION}_{TARGET_MIN}min.wav"
merged.export(out_file, format="wav")
print("Saved:", out_file)
