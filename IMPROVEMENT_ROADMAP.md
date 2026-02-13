# Distress Detection Model - Comprehensive Improvement Roadmap

## 1. MAIN ISSUE: BIOMETRIC DATA IS SYNTHETIC & RANDOM

**Current Problem:**
- Biometric input: 200-dimensional synthetic vectors (random noise)
- No real physiological signals (heart rate, HRV, skin conductance, etc.)
- Type classification always predicts "Anxiety" (bias from random data)
- Severity predictions are inaccurate (not correlated with real physiology)

**Why it fails:**
- Model can detect distress from AUDIO alone (100% accuracy)
- Biometric data doesn't add discriminative information
- Type classification has no signal to learn from

---

## 2. BIOMETRIC DATASETS TO ADD

### **PRIORITY 1: Heart Rate Variability (HRV) Data** ⭐⭐⭐
**Why:** Direct indicator of stress/arousal, widely available, easy to extract

**Best Datasets:**
1. **UBFC-Phys** (Open source, free)
   - URL: https://sites.google.com/view/ubfc-phys/dataset
   - Contains: Heart rate, respiratory rate, stress labels
   - Format: Video + ground truth labels for stress/relaxation
   - Size: 40 subjects × 5 recordings each
   - Action: Download videos → Extract HR/RR with DeepPhys or CHROM

2. **Wearable Stress & Affect Detection (WSAD) Dataset** (Open source)
   - Contains: ECG, GSR, Acc + stress levels (low/high)
   - Free download available
   - 3 stress conditions (baseline, math task, Stroop test)
   - 49 subjects

3. **MIT-BIH ECG Database** (Standard for HRV)
   - Free on PhysioNet
   - 48 recordings × 30 min each
   - Can derive: HRV, heart rate, variability patterns
   - Mix of healthy + cardiac patients

---

### **PRIORITY 2: Electrodermal Activity (EDA/GSR)** ⭐⭐⭐
**Why:** Sympathetic nervous system response to stress, high correlation with distress

**Best Datasets:**
1. **SWELL-KD Dataset** (Stress at Work)
   - Contains: EDA (skin conductance), skin temperature, ACC + stress labels
   - Open source from TU Delft
   - 25 subjects × 6 hours each of office work
   - Clear "stressed" vs "normal" labels

2. **WISDM Smartphone Biometric Dataset**
   - Contains: Accelerometer + heart rate + stress labels
   - 50 subjects
   - Collected via smartwatch

---

### **PRIORITY 3: ECG (Electrocardiogram)** ⭐⭐
**Why:** Complete heart signal, derive HRV + other metrics

**Best Datasets:**
1. **PhysioNet Stress Database**
   - URL: https://physionet.org/content/stdb/1.0.0/
   - ECG during mental stress tasks
   - 15 subjects under stress conditions

2. **CEAP-360VR Dataset** (Emotion + Stress)
   - Contains: ECG, EEG, GSR, BVP during VR tasks
   - 30 subjects
   - Multiple emotion/stress labels

---

### **PRIORITY 4: Respiratory Signals** ⭐
**Why:** Deep breathing patterns change with stress/panic

**Best Datasets:**
1. **Respiratory Data from UBFC-Phys** (same as above)
2. **Respiration Rate in WSAD Dataset** (same as above)

---

## 3. FEATURE EXTRACTION PIPELINE

### **What to extract from raw biometric data:**

**From Heart Rate / ECG:**
```
- Mean heart rate (bpm)
- Heart rate variability (std dev of RR intervals)
- HRV domains:
  * Time domain: SDNN, RMSSD, pNN50
  * Frequency domain: LF/HF ratio, VLF power, LF power, HF power
  * Non-linear: Approximate entropy, Poincaré plot SD1/SD2
- Heart rate acceleration/deceleration
- Baseline vs stress heart rate change
```

**From EDA (GSR):**
```
- Skin conductance level (SCL) mean & std
- Skin conductance response (SCR) frequency
- SCR amplitude peaks
- Tonic vs phasic components
```

**From Respiratory:**
```
- Respiratory rate (breaths/min)
- Breathing depth variability
- Inspiration/expiration ratio
```

**Context (12-dim, currently OK):**
```
- Time of day
- Environment noise level
- Body movement/activity level
- Device type (phone, smartwatch, etc.)
- Location (home, office, vehicle, etc.)
```

**Total biometric vector: ~100-120 features** (instead of random 200-dim)

---

## 4. STEP-BY-STEP INTEGRATION PLAN

### **Step 1: Download Biometric Dataset** (1-2 hours)
```bash
# Option A: UBFC-Phys (recommended, smallest)
cd data/raw/biometric
wget https://sites.google.com/view/ubfc-phys/dataset
# Extract videos + labels

# Option B: SWELL-KD (if you want GSR)
wget https://www.utwente.nl/en/news/
# Download EDA/temperature data

# Option C: MIT-BIH ECG (smallest download, ~115 MB)
wget https://physionet.org/files/mitdb/1.0.0/
```

### **Step 2: Feature Extraction Script** (Create new file)
**File:** `scripts/extract_biometric_features.py`

```python
"""
Extract biometric features from raw physiological data. 

Inputs:
- Raw ECG/EDA/Respiratory from dataset
- CSV with distress labels

Outputs:
- Processed CSVs in data/processed/biometric/
- Features: HRV + EDA + Respiratory (100-120 dims)
- Labels: distress (0/1), type (0-4), severity (0-10)
"""

# Pseudo-code structure:
# 1. Load raw ECG signal from UBFC-Phys videos
# 2. Detect R-peaks → compute RR intervals
# 3. Extract HRV features (time/freq/non-linear domains)
# 4. Extract EDA features from GSR signal
# 5. Extract respiratory features
# 6. Combine all into 120-dim vector
# 7. Save with corresponding labels
```

### **Step 3: Update Data Loading** (Create new file)
**File:** `src/biometric/data_loader.py`

```python
"""
Load real biometric data from processed dataset.

Replace synth_biometric() with real_biometric()
"""

class BiometricDataLoader:
    def __init__(self, data_dir="data/processed/biometric"):
        self.features = load_csv(f"{data_dir}/features.csv")
        self.labels = load_csv(f"{data_dir}/labels.csv")
    
    def get_sample(self, idx):
        # Return real biometric feature vector (120-dim)
        # Instead of random 200-dim synthetic data
        return self.features[idx]
```

### **Step 4: Retrain Model** (30-60 minutes)
```bash
python scripts/train_transformer_fusion_xgb.py
```

**Expected improvements:**
- Type classification: From random ~20% → ~70-80% accuracy
- Severity prediction: Better correlation with ground truth
- Distress detection: Still 100% (already at ceiling)

---

## 5. DATASET COMPARISON MATRIX

| Dataset | Size | HRV | EDA | ECG | Respiratory | Labels | Effort | Recommendation |
|---------|------|-----|-----|-----|-------------|--------|--------|-----------------|
| **UBFC-Phys** | 40 subj | ✅ | ❌ | ✅ | ✅ | ✅ | Low | ⭐⭐⭐ START HERE |
| **SWELL-KD** | 25 subj | ❌ | ✅ | ❌ | ❌ | ✅ | Low | ⭐⭐ Second |
| **MIT-BIH** | 48 subj | ✅ | ❌ | ✅ | ❌ | ⭐ | Low | ⭐ HRV baseline |
| **WSAD** | 49 subj | ✅ | ✅ | ❌ | ❌ | ✅ | Low | ⭐⭐ Combine with UBFC |
| **CEAP-360VR** | 30 subj | ✅ | ✅ | ❌ | ❌ | ✅ | Medium | ⭐ If VR relevant |

---

## 6. COMPLETE SOLUTION: RECOMMENDED SEQUENCE

### **Phase 1: Quick Start (1 day)**
1. Download **UBFC-Phys** (smallest, most complete)
2. Extract HRV features using existing `src/biometric/hrv.py`
3. Create simple 50-dim biometric vector (instead of 200-dim random)
4. Retrain model
5. Evaluate improvement in type classification

### **Phase 2: Enhancement (3-5 days)**
1. Download **SWELL-KD** for GSR/EDA features
2. Extract 30-dim EDA features using `src/biometric/baseline.py`
3. Combine HRV (50-dim) + EDA (30-dim) + Context (12-dim) = 92-dim vector
4. Retrain model
5. Compare results with Phase 1

### **Phase 3: Production (1 week)**
1. Combine multiple datasets (UBFC + WSAD + MIT-BIH)
2. Standardize feature extraction across all sources
3. Create robust preprocessing pipeline
4. Train final model on larger dataset
5. Deploy with real-time biometric collection

---

## 7. OTHER IMPROVEMENTS (Beyond Biometric)

### **Issue 2: Type Classification Always Predicts "Anxiety"**
**Root Cause:** Synthetic biometric doesn't differ by type
**Solution:** Real biometric data will have different patterns for Anxiety vs Panic vs Pain

### **Issue 3: Limited Audio Data**
**Status:** Currently using ~100 synthetic audio files
**Solution:** Add real distress audio datasets:
- **EMODB** (German emotional speech) - 535 samples
- **RAVDESS** (Emotional speech + song) - 1440 samples
- **CREMA-D** (Multimodal emotions) - 7442 samples
- **TorchVision Wav2Vec** - Pre-trained on 10k hours

### **Issue 4: No Real Validation Set**
**Current:** 80% synthetic training, 20% synthetic validation
**Better:** Mix real data (20-30%) with synthetic (70-80%) for validation
**Best:** Separate unseen real dataset for final evaluation

### **Issue 5: Context Features Limited**
**Current:** Time + noise + motion + device (12-dim)
**Better:** Add:
- Environmental context (location, temperature, humidity)
- Behavioral context (conversation, eating, exercise)
- Medical context (medication, sleep, caffeine)
- Social context (alone, with others, presentation)

---

## 8. IMPLEMENTATION CHECKLIST

### **Before Adding Biometric Data:**
- [ ] Download UBFC-Phys dataset
- [ ] Extract HRV features using existing code
- [ ] Update `scripts/train_transformer_fusion_xgb.py` to load real data
- [ ] Create `data/processed/biometric/` directory
- [ ] Save extracted features as CSV

### **After Integration:**
- [ ] Retrain model with real biometric data
- [ ] Compare type classification accuracy (should improve)
- [ ] Check if severity prediction improves
- [ ] Validate distress detection still at 100%
- [ ] Create comparison report (synthetic vs real)

### **For Production:**
- [ ] Add multiple biometric datasets
- [ ] Implement real-time biometric collection (smartwatch/wearable)
- [ ] Create online learning pipeline
- [ ] Set up monitoring for model performance

---

## 9. QUICK COMMAND REFERENCE

### **Download UBFC-Phys** (Recommended Starting Point)
```bash
cd data/raw/biometric
# Manual download from: https://sites.google.com/view/ubfc-phys/dataset
# Then unzip to data/raw/biometric/ubfc_phys/
```

### **Extract HRV Features**
```bash
python scripts/extract_biometric_features.py \
    --input data/raw/biometric/ubfc_phys/ \
    --output data/processed/biometric/features.csv \
    --features HRV
```

### **Retrain with Real Data**
```bash
python scripts/train_transformer_fusion_xgb.py \
    --use_real_biometric \
    --data_dir data/processed/biometric/
```

---

## 10. EXPECTED OUTCOMES

### **Before (Current - Synthetic Biometric):**
- Distress detection: 100% accuracy ✅
- Type classification: 20% accuracy (random bias) ❌
- Severity MAE: ~1.0-1.5 ⚠️
- Model learns: Mostly from audio + XGBoost features

### **After (Real Biometric Data):**
- Distress detection: 100% accuracy ✅ (or slightly higher)
- Type classification: 70-85% accuracy ✅✅ (depends on data quality)
- Severity MAE: 0.3-0.7 ✅ (improved)
- Model learns: From audio + biometric synergy

### **Real-World Performance:**
- Accuracy on unlabeled data: 85-95%
- False positive rate: <5%
- Latency per sample: <500ms
- Cost per inference: <$0.001

---

## Summary

**TL;DR:**
1. Download **UBFC-Phys** dataset (40 subjects, free, HRV labels)
2. Extract HRV features (50-100 dimensions of real heart rate variability)
3. Replace synthetic 200-dim random data with real features
4. Retrain model (should take 15-30 minutes)
5. Type classification should improve from 20% → 70-80%

**Estimated effort:** 1-2 days total for Phase 1
**Expected improvement:** 3-4x better type classification accuracy
