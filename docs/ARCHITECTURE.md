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

## Configuration Management

The system uses YAML-based configuration files located in `configs/`:

1. **model_config.yaml** - Defines model architecture:
   - Audio encoder (YAMNet) configuration
   - Biometric encoder (hidden layers, dropout)
   - Fusion layer architecture
   - Classifier configuration

2. **training_config.yaml** - Training hyperparameters:
   - Epochs, batch size, learning rate
   - Optimizer and scheduler settings
   - Loss function (weighted cross-entropy)
   - Data splits and augmentation
   - Checkpointing strategy

3. **deployment_config.yaml** - Deployment settings:
   - API server configuration (host, port, workers)
   - Inference settings (device, batch size)
   - Performance thresholds
   - Security settings (CORS, rate limiting)

The `Config` class in `src/utils/config.py` provides:
- YAML file loading with validation
- Multi-file configuration merging
- Dot-notation access (`config.get('key.subkey')`)
- Attribute-style access (`config.key.subkey`)

## Logging System

Structured logging is implemented in `src/utils/logger.py`:

- **Multiple Levels:** DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Console Handler:** Colored output for easy debugging
- **File Handler:** Rotating logs (10MB max, 5 backups) in `logs/` directory
- **Format:** Timestamps, log level, module name, function/line number, message

Log files are automatically rotated to prevent disk space issues.

## Technology Stack

- **ML Framework:** PyTorch 2.0+
- **Audio Processing:** librosa, YAMNet (TensorFlow Hub)
- **Biometric:** neurokit2, scipy
- **API:** FastAPI, uvicorn
- **Explainability:** SHAP
- **Testing:** pytest
- **Configuration:** PyYAML, custom Config class
- **Logging:** Python logging with custom handlers
