
from typing import Dict, List
import numpy as np
import pandas as pd
import joblib

import shap


class BiometricExplainer:
    def __init__(
        self,
        model_path: str,
        feature_names: List[str]
    ):

        self.model = joblib.load(model_path)
        self.feature_names = feature_names

        # TreeExplainer is optimal for XGBoost
        self.explainer = shap.TreeExplainer(self.model)

    # -------------------------------------------------
    # Explain a single biometric feature vector
    # -------------------------------------------------
    def explain(self, features: Dict[str, float]) -> Dict[str, float]:


        # Convert dict → DataFrame in correct order
        x = pd.DataFrame(
            [[features.get(f, 0.0) for f in self.feature_names]],
            columns=self.feature_names
        )

        # Compute SHAP values
        shap_values = self.explainer.shap_values(x)

        # Binary classifier → take class 1 (anomaly)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = shap_values[0]

        # Map feature → contribution
        explanation = {
            feature: float(value)
            for feature, value in zip(self.feature_names, shap_values)
        }

        return explanation

    # -------------------------------------------------
    # Convenience: ranked feature importance
    # -------------------------------------------------
    def explain_sorted(
        self,
        features: Dict[str, float],
        top_k: int = 10
    ) -> Dict[str, float]:

        explanation = self.explain(features)

        sorted_items = sorted(
            explanation.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return dict(sorted_items[:top_k])
