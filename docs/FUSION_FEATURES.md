# Fusion Layer Features Documentation

This document describes the new features implemented for the fusion layer module.

## Overview

The following features have been implemented as part of the fusion layer development:

1. **Test Suite (`tests/test_fusion.py`)** - Comprehensive tests for fusion layers and classifier
2. **Rule Engine (`src/fusion/rule_engine.py`)** - Rule-based system for checking extreme biometric values
3. **Explainer Module (`src/fusion/explainer.py`)** - Attention visualization and confidence calculation
4. **Demo Script (`scripts/test_fusion_features.py`)** - Complete demonstration of all features

## 1. Test Suite (`tests/test_fusion.py`)

### Features Tested:

- **Late Fusion Layer**
  - Initialization
  - Forward pass
  - Output shape verification

- **Attention Fusion Layer**
  - Initialization
  - Forward pass with attention weights
  - Output shape verification
  - Attention weights sum to 1 (softmax constraint)

- **Multi-Task Classifier**
  - Initialization
  - Forward pass with all three heads:
    - Binary distress detection (2 classes)
    - Severity regression (0-10 scale)
    - Distress type classification (5 classes)
  - Output shape verification

- **Rule Engine**
  - Initialization with configurable thresholds
  - Heart rate checks (normal, warning, critical)
  - Fall detection from acceleration data
  - Multiple alert aggregation

- **Attention Visualization**
  - Heatmap generation (requires matplotlib)
  - Single sample and batch support

- **Confidence Calculation**
  - Basic confidence calculation
  - Confidence with attention weights
  - Confidence with rule agreement

### Running Tests:

```bash
pytest tests/test_fusion.py -v
```

## 2. Rule Engine (`src/fusion/rule_engine.py`)

### Purpose

The `RuleEngine` class provides a rule-based system for checking extreme biometric values and triggering safety alerts.

### Key Features:

- **Heart Rate Monitoring**
  - Configurable thresholds (warning, critical, low)
  - Age-adjusted thresholds for children
  - Multiple alert levels

- **Fall Detection**
  - Based on acceleration magnitude
  - Configurable threshold (default: 20 m/s²)

- **Respiratory Rate Monitoring**
  - High and low threshold detection

- **Temperature Monitoring**
  - Fever detection (high temperature)
  - Hypothermia detection (low temperature)

- **Multi-Alert Support**
  - Check all biometric values at once
  - Aggregate alerts and determine overall severity

### Usage Example:

```python
from src.fusion.rule_engine import RuleEngine
import numpy as np

# Initialize rule engine
engine = RuleEngine(
    hr_critical_threshold=180.0,
    hr_warning_threshold=150.0
)

# Check heart rate
alert = engine.check_heart_rate(heart_rate=185.0)
if alert:
    print(f"Alert: {alert.message}")

# Check fall detection
acceleration = np.array([20.0, 18.0, 22.0])
fall_alert = engine.check_fall_detection(acceleration)

# Check all biometrics
alerts = engine.check_all(
    biometrics={
        'heart_rate': 185.0,
        'respiratory_rate': 45.0,
        'temperature': 39.0
    },
    acceleration=np.array([20.0, 18.0, 22.0])
)

# Get overall alert level
alert_level = engine.get_alert_level(alerts)
```

### Alert Levels:

- `NORMAL` - No alerts
- `WARNING` - Elevated values but not critical
- `CRITICAL` - Serious issue detected (e.g., fall)
- `EMERGENCY` - Extreme values requiring immediate attention

## 3. Explainer Module (`src/fusion/explainer.py`)

### Features:

#### A. Attention Weight Visualization

The `visualize_attention_weights()` function creates heatmaps showing which modality (audio, biometric, context) contributed most to the prediction.

**Features:**
- Heatmap visualization of attention weights across samples
- Bar plot showing average attention weights
- Support for single sample or batch
- Customizable modality names and sample labels
- Save to file option

**Usage:**

```python
from src.fusion.explainer import visualize_attention_weights
import torch

# Attention weights from AttentionFusion model
attention_weights = torch.tensor([
    [0.4, 0.5, 0.1],  # Sample 1: Audio=0.4, Bio=0.5, Context=0.1
    [0.2, 0.7, 0.1],  # Sample 2
])

fig = visualize_attention_weights(
    attention_weights,
    modalities=['Audio', 'Biometric', 'Context'],
    save_path='attention_heatmap.png'
)
```

**Requirements:** matplotlib and seaborn

#### B. Confidence Score Calculation

The `calculate_confidence()` function computes prediction confidence based on:

1. **Model Uncertainty** - Entropy of model outputs
2. **Modality Agreement** - Balance of attention weights
3. **Rule Agreement** - Alignment with rule-based system alerts

**Returns:**
- `distress_confidence` - Confidence in distress prediction (0-1)
- `severity_confidence` - Confidence in severity prediction (0-1)
- `type_confidence` - Confidence in distress type prediction (0-1)
- `overall_confidence` - Weighted average of all confidence scores
- `modality_agreement` - Agreement between modalities (0-1)
- `rule_agreement` - Agreement with rule-based system (0-1)

**Usage:**

```python
from src.fusion.explainer import calculate_confidence

model_outputs = {
    'distress_logits': torch.tensor([[2.0, 0.5]]),
    'severity': torch.tensor([[5.0]]),
    'type_logits': torch.tensor([[1.0, 0.5, 0.3, 0.2, 0.1]])
}

attention_weights = torch.tensor([[0.33, 0.33, 0.34]])

confidence = calculate_confidence(
    model_outputs=model_outputs,
    attention_weights=attention_weights,
    rule_alerts=alerts  # Optional list of Alert objects
)

print(f"Overall Confidence: {confidence['overall_confidence']:.3f}")
```

#### C. SHAP Explainer (Optional)

The `DistressExplainer` class provides SHAP-based model interpretability.

**Note:** Requires SHAP library and background data for initialization.

## 4. Demo Script (`scripts/test_fusion_features.py`)

A comprehensive demonstration script that tests all features:

```bash
python scripts/test_fusion_features.py
```

**What it demonstrates:**

1. Testing fusion layers with dummy inputs
2. Testing classifier output shapes
3. Visualizing attention weights (if matplotlib available)
4. Testing rule-based system with various scenarios
5. Calculating confidence scores

## Integration with Existing Code

All new features are integrated into the fusion module and can be imported from `src.fusion`:

```python
from src.fusion import (
    RuleEngine,
    AlertLevel,
    visualize_attention_weights,
    calculate_confidence,
    DistressExplainer
)
```

## Dependencies

### Required:
- `torch` - PyTorch for neural network operations
- `numpy` - Numerical operations

### Optional:
- `matplotlib` - For visualization (attention weights)
- `seaborn` - For better heatmaps
- `shap` - For SHAP-based explainability

Install optional dependencies:

```bash
pip install matplotlib seaborn shap
```

## File Structure

```
src/fusion/
├── rule_engine.py          # Rule-based system for extreme values
├── explainer.py            # Attention visualization and confidence
├── attention_fusion.py     # Attention-based fusion (existing)
├── late_fusion.py          # Late fusion layer (existing)
├── classifier.py           # Multi-task classifier (existing)
└── model.py                # Complete model (existing)

tests/
└── test_fusion.py          # Comprehensive test suite

scripts/
└── test_fusion_features.py # Demo script
```

## Notes

- All code follows the existing codebase style and conventions
- Only fusion-related code was modified (as per Intern 3's scope)
- Tests work with dummy data (no real dataset required)
- Visualization features gracefully handle missing dependencies
- Windows-compatible (no Unicode characters in print statements)

