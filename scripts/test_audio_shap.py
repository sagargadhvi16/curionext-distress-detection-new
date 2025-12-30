"""
End-to-end test for Audio SHAP explainability.
"""

print(">>> SCRIPT STARTED <<<", flush=True)

import torch

from src.fusion.model import DistressDetectionModel
from src.explainability.audio_wrapper import AudioOnlyWrapper
from src.explainability.audio_explainer import AudioExplainer
from src.explainability.feature_names import get_audio_feature_names


def main():
    FEATURE_DIM = 256
    BACKGROUND_SAMPLES = 5

    print("[INFO] Loading distress detection model...", flush=True)
    model = DistressDetectionModel()
    model.eval()

    print("[INFO] Creating audio-only wrapper...", flush=True)
    audio_model = AudioOnlyWrapper(model)

    print("[INFO] Preparing dummy audio features...", flush=True)
    background = torch.randn(BACKGROUND_SAMPLES, FEATURE_DIM)
    sample = torch.randn(1, FEATURE_DIM)

    print("[INFO] Loading feature names...", flush=True)
    feature_names = get_audio_feature_names()
    assert len(feature_names) == FEATURE_DIM

    print("[INFO] Initializing SHAP explainer...", flush=True)
    explainer = AudioExplainer(
        model=audio_model,
        background_data=background
    )

    print("[INFO] Computing SHAP values...", flush=True)
    result = explainer.explain(
        audio_features=sample,
        feature_names=feature_names
    )

    print("\n[RESULT] SHAP computation successful ✅", flush=True)
    print("Top 10 features:", result["feature_names"][:10], flush=True)

    print("\n[OK] Audio SHAP test completed successfully!", flush=True)


if __name__ == "__main__":
    main()
