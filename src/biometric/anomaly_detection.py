"""
Supervised Biometric Anomaly Detection (Child)

Model:
- XGBoost Classifier (age-aware, uncertainty-aware)

Labels:
0 = Normal
1 = Stress / Anomaly
-1 = Uncertain (borderline)
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ================================
# CONFIG
# ================================
DATA_PATH = "synthetic_biometric_dataset.csv"
MODEL_PATH = "models/biometric_anomaly_xgb.pkl"

LABEL_COL = "anomaly"
DROP_COLS = ["anomaly", "state_label", "activity_label"]

# Uncertainty thresholds
LOW_CONF = 0.30
HIGH_CONF = 0.70

# ================================
# LOAD DATA
# ================================
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Train only on clean samples
    df = df[df["is_ambiguous"] == 0]

    y = df[LABEL_COL].astype(int)

    X = df.drop(columns=DROP_COLS, errors="ignore")

    # One-hot encode age bucket
    if "age_bucket" in X.columns:
        X = pd.get_dummies(X, columns=["age_bucket"])

    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"[INFO] Clean dataset shape: {df.shape}")
    print(f"[INFO] Feature count: {X.shape[1]}")
    print(f"[INFO] Class distribution:\n{y.value_counts()}")

    return X, y

# ================================
# TRAIN MODEL
# ================================
def train_model():
    print("[INFO] Loading synthetic data...")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("[INFO] Training XGBoost...")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=0.5,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ================================
    # UNCERTAINTY-AWARE EVALUATION
    # ================================
    probs = model.predict_proba(X_test)[:, 1]

    confident_mask = (probs < LOW_CONF) | (probs > HIGH_CONF)
    confident_probs = probs[confident_mask]
    confident_labels = y_test.iloc[confident_mask]

    confident_preds = (confident_probs > HIGH_CONF).astype(int)

    coverage = confident_mask.mean()
    acc_confident = accuracy_score(confident_labels, confident_preds)

    print("\n[INFO] Uncertainty-aware evaluation")
    print(f"Coverage (confident predictions): {coverage:.2%}")
    print(f"Accuracy on confident predictions: {acc_confident:.3f}")

    print("\nClassification Report (confident only):\n")
    print(classification_report(confident_labels, confident_preds, digits=3))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

# ================================
# INFERENCE
# ================================
def detect_biometric_anomaly(features: dict) -> dict:
    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([features])

    if "age_bucket" in df.columns:
        df = pd.get_dummies(df, columns=["age_bucket"])

    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    prob = float(model.predict_proba(df)[0][1])

    if prob < LOW_CONF:
        decision = 0
        label = "normal"
    elif prob > HIGH_CONF:
        decision = 1
        label = "anomaly"
    else:
        decision = -1
        label = "uncertain"

    return {
        "decision": label,
        "anomaly_score": round(prob, 3),
        "reason": "XGBoost with uncertainty handling"
    }

# ================================
# RUN
# ================================
if __name__ == "__main__":
    train_model()
