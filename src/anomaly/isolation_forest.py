"""
Isolation Forest Anomaly Detection Module
==========================================
Detects anomalous transactions using Isolation Forest. Anomalous
transactions require fewer random partitions to isolate, yielding
higher anomaly scores.
"""

import numpy as np
import pandas as pd
from typing import Optional
import joblib
import os

from sklearn.ensemble import IsolationForest

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detection.

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalies in the data.
    n_estimators : int
        Number of trees in the forest.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        contamination: float = 0.035,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )
        self.is_fitted = False

    def fit_predict(self, X: np.ndarray) -> dict:
        """
        Fit Isolation Forest and compute anomaly scores.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        dict with keys:
            - 'anomaly_label': int array, 1=normal, -1=anomaly
            - 'anomaly_score': float array, continuous score (higher = more anomalous)
            - 'anomaly_score_normalized': float array, normalized to [0, 1]
        """
        logger.info("=" * 60)
        logger.info("ISOLATION FOREST ANOMALY DETECTION")
        logger.info("=" * 60)
        logger.info(f"  Input shape: {X.shape}")
        logger.info(f"  n_estimators: {self.n_estimators}")
        logger.info(f"  contamination: {self.contamination}")

        # Handle NaN/inf
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit and predict
        anomaly_labels = self.model.fit_predict(X_clean)
        self.is_fitted = True

        # Raw anomaly scores (negative = more anomalous in sklearn)
        raw_scores = self.model.decision_function(X_clean)

        # Invert and normalize to [0, 1] where 1 = most anomalous
        anomaly_scores = -raw_scores  # Now higher = more anomalous
        score_min = anomaly_scores.min()
        score_max = anomaly_scores.max()
        score_range = score_max - score_min

        if score_range > 0:
            normalized_scores = (anomaly_scores - score_min) / score_range
        else:
            normalized_scores = np.zeros_like(anomaly_scores)

        results = {
            "anomaly_label": anomaly_labels,
            "anomaly_score": anomaly_scores,
            "anomaly_score_normalized": normalized_scores,
        }

        n_anomalies = (anomaly_labels == -1).sum()
        logger.info(f"  Anomalies detected: {n_anomalies} ({n_anomalies / len(X) * 100:.1f}%)")
        logger.info(f"  Score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")

        return results

    def predict(self, X: np.ndarray) -> dict:
        """
        Predict anomaly scores for new data.

        Parameters
        ----------
        X : np.ndarray
            New feature matrix.

        Returns
        -------
        dict
            Anomaly labels and scores.
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector has not been fitted.")

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        anomaly_labels = self.model.predict(X_clean)
        raw_scores = self.model.decision_function(X_clean)
        anomaly_scores = -raw_scores

        score_min = anomaly_scores.min()
        score_max = anomaly_scores.max()
        score_range = score_max - score_min

        if score_range > 0:
            normalized_scores = (anomaly_scores - score_min) / score_range
        else:
            normalized_scores = np.zeros_like(anomaly_scores)

        return {
            "anomaly_label": anomaly_labels,
            "anomaly_score": anomaly_scores,
            "anomaly_score_normalized": normalized_scores,
        }

    def save(self, filepath: str) -> None:
        """Save fitted model."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"  Isolation Forest saved to: {filepath}")

    def load(self, filepath: str) -> None:
        """Load fitted model."""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"  Isolation Forest loaded from: {filepath}")
