# XGBoost Audio Baseline (Coarse Distress Classification)

This module implements a classical audio baseline for distress detection using
hand-crafted audio features and an XGBoost classifier.

## Task Setup

- Training data: `data/CurioNext_Audio/train`
  - 6 emotion folders: anger, abuse, scream, fear, cry, neutral
  - From each emotion: 10 audio files
  - From each audio: first 10 windows of 10 seconds
  - Total training samples: 600 windows

- Label mapping (coarse classes):
  - anger, abuse, scream → high_arousal
  - fear, cry → distress
  - neutral → neutral

## Features

- MFCC
- Mel Spectrogram
- Chroma
- Spectral Contrast
- Tonnetz
- Statistical pooling (mean, std, min, max)

## Model

- XGBoost multi-class classifier (3 classes)
- Trained on window-level features
- Model artifacts saved as `.joblib`

## Evaluation

- Test data: `data/CurioNext_Audio/test` (wav + json)
- Window-level accuracy (coarse): ~0.49
- Sequence-level accuracy (majority vote): ~0.90

Window accuracy measures performance on individual 10s segments,
while sequence accuracy measures correctness of the dominant emotion
over the full audio.

## Notes

- This is a classical baseline, not the final model
- Intended for comparison and fusion with SSL-based models

## Artifacts
Pre-trained XGBoost model, scaler, and label encoder are provided in `artifacts/`
for quick testing and fusion. Retraining using `train_xgb.py` is recommended
for any configuration changes.
