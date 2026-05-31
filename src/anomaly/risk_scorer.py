"""
Risk Scorer Module
==================
Combines multiple anomaly signals (Isolation Forest, HDBSCAN outlier scores,
centroid distances, behavioral deviations) into a single composite risk score.
Weights are calibrated using logistic regression.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskScorer:
    """
    Composite risk score calculator.

    Combines multiple fraud signals into a single probabilistic risk
    score. Weights are initially set from config but can be calibrated
    using logistic regression on labeled data.

    Parameters
    ----------
    initial_weights : dict
        Initial weights for each signal component.
    """

    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        self.weights = initial_weights or {
            "anomaly_score": 0.25,
            "outlier_probability": 0.20,
            "centroid_distance": 0.15,
            "amount_deviation": 0.15,
            "velocity_score": 0.10,
            "cluster_confidence_inv": 0.15,
        }
        self.calibrator = LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced"
        )
        self.signal_scaler = MinMaxScaler()
        self.is_calibrated = False
        self.calibrated_weights: Optional[np.ndarray] = None

    def compute_risk_score(
        self,
        anomaly_score: np.ndarray,
        outlier_score: np.ndarray,
        cluster_probability: np.ndarray,
        centroid_distance: np.ndarray,
        cluster_fraud_rate: np.ndarray,
        amount_zscore: Optional[np.ndarray] = None,
        velocity_score: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute composite risk score from multiple signals.

        Parameters
        ----------
        anomaly_score : np.ndarray
            Normalized Isolation Forest anomaly score [0, 1].
        outlier_score : np.ndarray
            HDBSCAN GLOSH outlier score.
        cluster_probability : np.ndarray
            HDBSCAN cluster membership probability.
        centroid_distance : np.ndarray
            Distance from transaction to its cluster centroid.
        cluster_fraud_rate : np.ndarray
            Historical fraud rate of the assigned cluster.
        amount_zscore : np.ndarray, optional
            Transaction amount z-score (absolute value used).
        velocity_score : np.ndarray, optional
            Transaction velocity anomaly indicator.

        Returns
        -------
        dict with keys:
            - 'risk_score': float array, composite risk [0, 1]
            - 'risk_signals': pd.DataFrame of individual normalized signals
            - 'risk_category': str array, 'LOW'/'MEDIUM'/'HIGH'/'CRITICAL'
        """
        logger.info("=" * 60)
        logger.info("RISK SCORE COMPUTATION")
        logger.info("=" * 60)

        n = len(anomaly_score)

        # Normalize each signal to [0, 1]
        signals = {
            "anomaly_score": self._normalize(anomaly_score),
            "outlier_probability": self._normalize(outlier_score),
            "centroid_distance": self._normalize(centroid_distance),
            "cluster_fraud_rate": np.clip(cluster_fraud_rate, 0, 1),
            "cluster_confidence_inv": 1.0 - np.clip(cluster_probability, 0, 1),
        }

        if amount_zscore is not None:
            signals["amount_deviation"] = self._normalize(np.abs(amount_zscore))
        else:
            signals["amount_deviation"] = np.zeros(n)

        if velocity_score is not None:
            signals["velocity_score"] = self._normalize(velocity_score)
        else:
            signals["velocity_score"] = np.zeros(n)

        # Compute weighted composite score
        if self.is_calibrated and self.calibrated_weights is not None:
            # Use calibrated weights from logistic regression
            signal_matrix = np.column_stack([signals[k] for k in sorted(signals.keys())])
            risk_score = self.calibrator.predict_proba(signal_matrix)[:, 1]
        else:
            # Use initial weights
            risk_score = np.zeros(n)
            for signal_name, signal_values in signals.items():
                weight = self.weights.get(signal_name, 0.0)
                risk_score += weight * signal_values

            # Clip to [0, 1]
            risk_score = np.clip(risk_score, 0, 1)

        # Categorize risk levels
        risk_category = self._categorize_risk(risk_score)

        # Create signals DataFrame
        risk_signals = pd.DataFrame(signals)

        logger.info(f"  Risk score statistics:")
        logger.info(f"    Mean: {risk_score.mean():.4f}")
        logger.info(f"    Std:  {risk_score.std():.4f}")
        logger.info(f"    Min:  {risk_score.min():.4f}")
        logger.info(f"    Max:  {risk_score.max():.4f}")
        logger.info(f"  Risk categories:")
        for cat in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            count = (risk_category == cat).sum()
            logger.info(f"    {cat}: {count} ({count / n * 100:.1f}%)")

        return {
            "risk_score": risk_score,
            "risk_signals": risk_signals,
            "risk_category": risk_category,
        }

    def calibrate(
        self,
        signals: pd.DataFrame,
        labels: np.ndarray,
    ) -> None:
        """
        Calibrate risk score weights using logistic regression.

        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame of individual risk signals.
        labels : np.ndarray
            True fraud labels (0/1).
        """
        logger.info("  Calibrating risk score weights with logistic regression...")

        X = np.nan_to_num(signals.values, nan=0.0, posinf=0.0, neginf=0.0)
        y = labels.values if hasattr(labels, "values") else labels

        # Fit calibrator
        self.calibrator.fit(X, y)
        self.is_calibrated = True

        # Extract calibrated weights
        self.calibrated_weights = self.calibrator.coef_[0]

        # Log learned weights
        feature_names = signals.columns.tolist()
        weight_pairs = list(zip(feature_names, self.calibrated_weights))
        weight_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        logger.info("  Calibrated weights (by importance):")
        for name, weight in weight_pairs:
            logger.info(f"    {name}: {weight:+.4f}")

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        arr = np.asarray(arr, dtype=float)
        # HDBSCAN outlier scores are NaN for non-outlier points; treat as 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr_min = arr.min()
        arr_max = arr.max()
        rng = arr_max - arr_min

        if rng > 0:
            return (arr - arr_min) / rng
        return np.zeros_like(arr)

    @staticmethod
    def _categorize_risk(risk_score: np.ndarray) -> np.ndarray:
        """Categorize risk score into levels."""
        categories = np.full(len(risk_score), "LOW", dtype="<U10")
        categories[risk_score >= 0.25] = "MEDIUM"
        categories[risk_score >= 0.50] = "HIGH"
        categories[risk_score >= 0.75] = "CRITICAL"
        return categories

    def save(self, filepath: str) -> None:
        """Save calibrated risk scorer."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_data = {
            "calibrator": self.calibrator,
            "weights": self.weights,
            "calibrated_weights": self.calibrated_weights,
            "is_calibrated": self.is_calibrated,
        }
        joblib.dump(save_data, filepath)
        logger.info(f"  Risk scorer saved to: {filepath}")

    def load(self, filepath: str) -> None:
        """Load calibrated risk scorer."""
        save_data = joblib.load(filepath)
        self.calibrator = save_data["calibrator"]
        self.weights = save_data["weights"]
        self.calibrated_weights = save_data["calibrated_weights"]
        self.is_calibrated = save_data["is_calibrated"]
        logger.info(f"  Risk scorer loaded from: {filepath}")
