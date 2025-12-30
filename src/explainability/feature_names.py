"""
Centralized audio feature name mapping for SHAP explanations.
"""

def get_audio_feature_names() -> list[str]:
    names = []

    # MFCCs
    for i in range(40):
        names.append(f"MFCC_{i+1}")

    # MFCC deltas
    for i in range(40):
        names.append(f"MFCC_delta_{i+1}")

    # MFCC delta-delta
    for i in range(40):
        names.append(f"MFCC_delta2_{i+1}")

    # Prosodic features
    prosodic = [
        "pitch_mean", "pitch_std",
        "energy_mean", "energy_std",
        "jitter", "shimmer",
        "voicing_rate", "pause_ratio",
        "hnr", "intensity_mean",
        "intensity_std", "f0_min",
        "f0_max", "f0_range",
        "speech_rate", "silence_ratio"
    ]
    names.extend(prosodic)

    # Spectral features
    spectral = [
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "zcr_mean", "zcr_std",
        "spectral_flatness", "spectral_contrast",
        "spectral_entropy", "spectral_flux",
        "chroma_mean", "chroma_std",
        "rms_mean", "rms_std"
    ]
    names.extend(spectral)

    # Padding / learned features
    remaining = 256 - len(names)
    for i in range(remaining):
        names.append(f"learned_feature_{i+1}")

    return names
