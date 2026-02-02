import numpy as np
import whisper
import re
from collections import Counter

from audio_experiments.xgb_baseline.predict_xgb import predict_xgb


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
MAX_ASR_CHARS = 300   # Prevent UI / LLM pollution


# ---------------------------------------------------------------------
# Load Whisper ONCE
# ---------------------------------------------------------------------
_WHISPER_MODEL = whisper.load_model("base")  # CPU-safe


# ---------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------
def run_asr(audio_chunk: np.ndarray, sr: int = 16000) -> str:
    """
    Run Whisper ASR on an audio chunk.
    Returns raw transcript string (may be empty or noisy).
    """
    audio_chunk = audio_chunk.astype(np.float32)

    result = _WHISPER_MODEL.transcribe(
        audio_chunk,
        fp16=False,
        language=None,
        temperature=0.0,
        condition_on_previous_text=False,
        beam_size=5
    )

    return result.get("text", "").strip()


# ---------------------------------------------------------------------
# ASR QUALITY FILTERS
# ---------------------------------------------------------------------
def is_valid_speech(text: str) -> bool:
    """
    Filters obvious ASR garbage:
    - Too short
    - Low alphabetic ratio
    - Elongated characters
    - Excessive repetition
    """
    if not text:
        return False

    text = text.strip()

    # 1. Minimum length
    if len(text) < 8:
        return False

    # 2. Alphabetic content check
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars < 5:
        return False

    alpha_ratio = alpha_chars / max(len(text), 1)
    if alpha_ratio < 0.5:
        return False

    # 3. Reject elongated characters (AAAAA, ああああ, EEEEE)
    if re.search(r"(.)\1{5,}", text):
        return False

    # 4. Reject excessive word repetition
    words = text.lower().split()
    if len(words) >= 6:
        most_common = Counter(words).most_common(1)[0][1]
        if most_common / len(words) > 0.5:
            return False

    # 5. Reject very short repeated tokens ("yeah yeah yeah")
    if len(set(words)) <= 2 and len(words) > 5:
        return False

    return True


def is_mostly_readable(text: str, min_ascii_ratio: float = 0.4) -> bool:
    """
    Prevents Unicode garbage spam from polluting downstream systems.
    Allows English / Hinglish, blocks noisy scripts.
    """
    if not text:
        return False

    ascii_chars = sum(ord(c) < 128 for c in text)
    return (ascii_chars / len(text)) >= min_ascii_ratio


# ---------------------------------------------------------------------
# MAIN INFERENCE WRAPPER
# ---------------------------------------------------------------------
def predict_xgb_with_asr(audio_chunk: np.ndarray, sr: int = 16000):
    """
    Audio inference wrapper.

    Pipeline:
    1. XGBoost → coarse emotion
    2. If high_arousal / distress:
       - Run ASR
       - Validate ASR text
       - Tag as verbal / non-verbal distress

    Returns:
        {
          audio_label: str,
          audio_confidence: float,
          distress_type: str,
          asr_text: Optional[str]
        }
    """

    # Step 1: XGBoost prediction
    label, confidence = predict_xgb(audio_chunk, sr)

    distress_type = "neutral"
    asr_text = None

    # Step 2: Conditional ASR
    if label in ["high_arousal", "distress"]:
        raw_text = run_asr(audio_chunk, sr)

        if is_valid_speech(raw_text) and is_mostly_readable(raw_text):
            distress_type = "verbal_distress"

            # Truncate long ASR output (demo + UI safe)
            if len(raw_text) > MAX_ASR_CHARS:
                asr_text = raw_text[:MAX_ASR_CHARS] + "..."
            else:
                asr_text = raw_text
        else:
            distress_type = "non_verbal_distress"
            asr_text = None

    return {
        "audio_label": label,
        "audio_confidence": float(confidence),
        "distress_type": distress_type,
        "asr_text": asr_text
    }
