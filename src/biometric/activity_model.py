"""
ML-based Activity Level Classification using Accelerometer data

Classes:
- Sedentary
- Light
- Moderate
- Vigorous

Dataset:
- Accelerometer (acc_x, acc_y, acc_z)
- Activity labels (0–5)
"""

import numpy as np
import pandas as pd
import joblib
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Reuse your existing accelerometer feature extractor
from src.biometric.accelerometer import extract_accelerometer_features


# ---------------------------------------------------------------------
# 1️⃣ Load raw activity data
# ---------------------------------------------------------------------
def load_activity_data(csv_path: str) -> pd.DataFrame:
    """
    Load accelerometer activity dataset from CSV.
    """
    df = pd.read_csv(csv_path)
    return df


# ---------------------------------------------------------------------
# 2️⃣ Map numeric activity labels → activity levels
# ---------------------------------------------------------------------
def map_activity_label(activity_id: int) -> str:
    """
    Map dataset activity IDs to activity levels.

    0 = Downstairs
    1 = Jogging
    2 = Sitting
    3 = Standing
    4 = Upstairs
    5 = Walking
    """

    if activity_id in [2, 3]:          # Sitting, Standing
        return "sedentary"
    elif activity_id in [5]:           # Walking
        return "light"
    elif activity_id in [0, 4]:        # Upstairs, Downstairs
        return "moderate"
    elif activity_id in [1]:           # Jogging
        return "vigorous"
    else:
        return "unknown"


# ---------------------------------------------------------------------
# 3️⃣ Window data + extract features
# ---------------------------------------------------------------------
def build_feature_dataset(
    df: pd.DataFrame,
    window_size: int = 250,   # ~5 sec if ~50 Hz
    step_size: int = 250
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw accelerometer data into feature vectors.
    """

    X = []
    y = []

    for start in range(0, len(df) - window_size, step_size):
        window = df.iloc[start:start + window_size]

        accel_data = window[["acc_x", "acc_y", "acc_z"]].values
        activity_id = int(window["Activity"].mode()[0])

        features = extract_accelerometer_features(accel_data)
        label = map_activity_label(activity_id)

        if label == "unknown":
            continue

        X.append(list(features.values()))
        y.append(label)

    return np.array(X), np.array(y)


# ---------------------------------------------------------------------
# 4️⃣ Train ML model
# ---------------------------------------------------------------------
def train_activity_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """
    Train Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------
# 5️⃣ Main pipeline
# ---------------------------------------------------------------------
def main():
    data_path = "data/raw/activity/activity_data.csv"

    print("[INFO] Loading dataset...")
    df = load_activity_data(data_path)

    print("[INFO] Building feature dataset...")
    X, y = build_feature_dataset(df)

    print("[INFO] Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("[INFO] Training activity classifier...")
    model = train_activity_model(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[INFO] Saving model...")
    joblib.dump(model, "models/activity_model.pkl")

    print("✅ Activity classification training complete.")




def predict_activity_from_accel(
    accel_data: np.ndarray,
    model_path: str = "models/activity_model.pkl"
) -> str:
    """
    Predict activity level from raw accelerometer window.
    
    accel_data: shape (n_samples, 3)
    """
    # Load trained model
    model = joblib.load(model_path)

    # Extract features using your existing pipeline
    features = extract_accelerometer_features(accel_data)

    # Convert dict → list
    feature_vector = np.array(list(features.values())).reshape(1, -1)

    # Predict
    prediction = model.predict(feature_vector)[0]
    return prediction

# ---------------------------------------------------------------------
# 6️⃣ Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
