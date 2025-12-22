"""
Biometric Anomaly Detection (ML-Based)

Approach:
- Unsupervised Autoencoder
- Trained on NORMAL HRV patterns (low NASA score)
- High reconstruction error => anomaly

Deliverable:
detect_biometric_anomaly(features, baseline)

Dataset:
data/raw/kaggle/hrv_mwl_30.xlsx
"""

from typing import Dict, Any, List
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =====================================================
# 🧠 Autoencoder Model
# =====================================================
class HRVAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# =====================================================
# 🚨 ML-Based Anomaly Detector
# =====================================================
class BiometricAnomalyDetector:
    """
    Autoencoder-based anomaly detector for HRV.
    """

    def __init__(self):
        self.feature_keys = ["hr", "sdnn", "rmssd", "lf", "hf", "lfhf"]
        self.model = HRVAutoencoder(input_dim=len(self.feature_keys))
        self.threshold: float | None = None

    # -------------------------------------------------
    # Load Kaggle dataset (CSV / XLSX)
    # -------------------------------------------------
    def load_dataset(
        self,
        data_dir: str,
        nasa_threshold: float = 60.0
    ) -> List[Dict[str, float]]:
        """
        Load NORMAL HRV samples from Kaggle dataset.
        """

        rows: List[Dict[str, float]] = []

        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)

            # Load file
            if file.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file.endswith(".xlsx"):
                df = pd.read_excel(file_path)
            else:
                continue

            required_cols = set(self.feature_keys + ["nasa"])
            if not required_cols.issubset(df.columns):
                print(f"[WARN] Skipping {file}, missing columns")
                continue

            # Filter NORMAL (low stress)
            normal_df = df[df["nasa"] <= nasa_threshold]

            print(
                f"[INFO] {file}: "
                f"{len(normal_df)} normal samples found"
            )

            rows.extend(
                normal_df[self.feature_keys].to_dict("records")
            )

        if len(rows) == 0:
            raise RuntimeError(
                "No normal HRV data found. "
                "Increase nasa_threshold or check dataset."
            )

        return rows

    # -------------------------------------------------
    # Train autoencoder
    # -------------------------------------------------
    def fit(self, normal_rows: List[Dict[str, float]]) -> None:
        X = np.array(
            [[row[k] for k in self.feature_keys] for row in normal_rows],
            dtype=np.float32
        )

        X_tensor = torch.tensor(X)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(80):
            optimizer.zero_grad()
            recon = self.model(X_tensor)
            loss = loss_fn(recon, X_tensor)
            loss.backward()
            optimizer.step()

        # Compute anomaly threshold
        with torch.no_grad():
            recon = self.model(X_tensor)
            errors = torch.mean((recon - X_tensor) ** 2, dim=1).numpy()

        self.threshold = float(np.mean(errors) + 2 * np.std(errors))

    # -------------------------------------------------
    # Detect anomaly (core ML logic)
    # -------------------------------------------------
    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if self.threshold is None:
            raise RuntimeError("Detector not trained. Call fit() first.")

        x = np.array(
            [[features.get(k, 0.0) for k in self.feature_keys]],
            dtype=np.float32
        )

        x_tensor = torch.tensor(x)

        with torch.no_grad():
            recon = self.model(x_tensor)
            error = torch.mean((recon - x_tensor) ** 2).item()

        anomaly_score = min(error / self.threshold, 1.0)

        return {
            "anomaly_detected": error > self.threshold,
            "anomaly_score": round(anomaly_score, 2),
            "reason": "ML-based HRV anomaly"
        }


# =====================================================
# ✅ DELIVERABLE FUNCTION (REQUIRED BY TASK)
# =====================================================
def detect_biometric_anomaly(
    features: Dict[str, Any],
    baseline: BiometricAnomalyDetector
) -> Dict[str, Any]:
    """
    Deliverable-compliant anomaly detection function.

    Args:
        features: Unified biometric feature dictionary
        baseline: Trained BiometricAnomalyDetector (ML baseline)

    Returns:
        Anomaly detection result
    """
    return baseline.detect(features)


# =====================================================
# 🧪 Local Test (REAL DATA)
# =====================================================
if __name__ == "__main__":

    DATA_PATH = "data/raw/kaggle"

    detector = BiometricAnomalyDetector()

    print("[INFO] Loading dataset...")
    normal_data = detector.load_dataset(DATA_PATH)

    print(f"[INFO] Training on {len(normal_data)} normal samples...")
    detector.fit(normal_data)

    # Example stressed HRV
    test_sample = {
        "hr": 80,
        "sdnn": 40.5,
        "rmssd": 35.5,
        "lf": 69.77,
        "hf": 30.14,
        "lfhf": 2.31
    }

    result = detect_biometric_anomaly(
        features=test_sample,
        baseline=detector
    )

    print("\nML-Based Anomaly Detection Result:\n", result)
