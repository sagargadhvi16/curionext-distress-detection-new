# System Architecture

## Overview

The CurioNext Child Distress Detection System is a multi-modal AI system that fuses audio and biometric signals to detect distress in real-time.

## High-Level Architecture

```
┌─────────────────┐
│  Audio Input    │
│  (16kHz WAV)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Audio Preprocessing │
│  - Normalize         │
│  - Trim Silence      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  YAMNet Encoder     │
│  (Pretrained)       │
│  Output: 1024-dim   │
└──────────┬──────────┘
           │
           │                  ┌──────────────────┐
           │                  │ Biometric Input  │
           │                  │ - HRV (RR)       │
           │                  │ - Accelerometer  │
           │                  └────────┬─────────┘
           │                           │
           │                           ▼
           │                  ┌──────────────────┐
           │                  │ Feature Extract  │
           │                  │ - Time domain    │
           │                  │ - Freq domain    │
           │                  │ - Movement stats │
           │                  └────────┬─────────┘
           │                           │
           │                           ▼
           │                  ┌──────────────────┐
           │                  │ Biometric Encoder│
           │                  │ Output: 64-dim   │
           │                  └────────┬─────────┘
           │                           │
           └───────────┬───────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Late Fusion   │
              │  (Concat)      │
              │  1024 + 64     │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  Fusion Layers │
              │  512 → 256     │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  Classifier    │
              │  Binary Output │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  Distress /    │
              │  No Distress   │
              └────────────────┘
```

## Component Details

### 1. Audio Pipeline (Intern 1)

**Input:** 16kHz WAV audio files

**Components:**
- `preprocessing.py`: Audio loading, normalization, silence trimming
- `features.py`: MFCC, spectral features extraction (optional)
- `encoder.py`: YAMNet embedding extraction

**Output:** 1024-dimensional audio embeddings

### 2. Biometric Pipeline (Intern 2)

**Input:** HRV (RR intervals) + 3-axis accelerometer data

**Components:**
- `hrv.py`: Time/frequency domain HRV features
- `accelerometer.py`: Movement features
- `encoder.py`: Neural encoder for biometric features

**Output:** 64-dimensional biometric embeddings

### 3. Fusion & Backend (Intern 3)

**Components:**
- `late_fusion.py`: Concatenation-based fusion
- `classifier.py`: Binary distress classifier
- `explainer.py`: SHAP-based interpretability
- `api/`: FastAPI REST endpoints

**Output:** Distress probability + explanation

## Data Flow

1. Audio → Preprocess → YAMNet → 1024-dim embedding
2. HRV + Accel → Feature Extraction → Neural Encoder → 64-dim embedding
3. Concat embeddings → Fusion layers → Classifier → Prediction

## Model Training Flow

```
Data Loading → Augmentation → Batch Processing →
Forward Pass → Loss Computation → Backpropagation →
Optimizer Step → Validation → Checkpoint Saving
```

## API Endpoints

### `POST /predict`
- **Input:** Audio file + HRV data + Accelerometer data
- **Output:** Distress probability, prediction, confidence, inference time

### `GET /health`
- **Output:** Service status, model loaded, version

## Performance Requirements

- **Inference Time:** < 500ms
- **Model Size:** < 100MB
- **Accuracy:** > 90%
- **False Negative Rate:** < 5% (critical)

## Technology Stack

- **ML Framework:** PyTorch 2.0+
- **Audio Processing:** librosa, YAMNet (TensorFlow Hub)
- **Biometric:** neurokit2, scipy
- **API:** FastAPI, uvicorn
- **Explainability:** SHAP
- **Testing:** pytest
