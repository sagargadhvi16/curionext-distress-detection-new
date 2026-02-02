# Fusion Pipeline - Endpoints & UI Outputs (as per current code)

## 1. Multi-Modal Input

**Endpoint**: `/fusion/upload`

**Output**: audio features (775-dim) + biometric (200-dim) + context (12-dim) + status

**UI**: 3 status cards, dimension summary

---

## 2. Fusion Prediction

**Endpoint**: `/fusion/predict`

**Output**: distress label (Yes/No) + confidence + type (Anxiety/Depression/Panic) + severity (0-10)

**UI**: alert badge, confidence gauge, severity slider, type label

---

## 3. Model Metrics

**Endpoint**: `/fusion/metrics`

**Output**: distress accuracy (100%) + avg confidence + inference time (145ms)

**UI**: metric cards, speed indicator

---

## 4. Training History

**Endpoint**: `/fusion/training/history`

**Output**: loss per epoch (10 epochs) + accuracy progression

**UI**: line chart (loss over epochs), accuracy badge

---

## 5. ROC Curve

**Endpoint**: `/fusion/metrics/roc`

**Output**: ROC curve data + AUC score (1.00)

**UI**: ROC plot, AUC badge

---

## 6. Modality Contribution

**Endpoint**: `/fusion/modality-weights`

**Output**: audio (~45%), biometric (~38%), context (~17%)

**UI**: pie chart, percentage labels

---

## 7. Pipeline Status

**Endpoint**: `/fusion/pipeline/status`

**Output**: step completion flags (audio → biometric → context → fusion)

**UI**: progress stepper, terminal log
