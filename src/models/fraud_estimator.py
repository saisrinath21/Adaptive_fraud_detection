"""
Supervised Fraud Estimator (LightGBM)
=====================================
Trains a gradient-boosted classifier on risk / behavioral features and
outputs fraud probabilities for the RL decision layer.
"""

import os
from typing import Dict, List, Optional

import joblib
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FraudEstimator:
    """
    LightGBM-based supervised fraud probability estimator.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Learning rate.
    num_leaves : int
        Max leaves per tree.
    class_weight : str or dict
        ``'balanced'`` uses scale_pos_weight from training fraud rate.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 64,
        max_depth: int = -1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        class_weight: str = "balanced",
        random_state: int = 42,
        early_stopping_rounds: int = 50,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.class_weight = class_weight
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Train LightGBM on labeled transactions."""
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for FraudEstimator. "
                "Install with: pip install lightgbm"
            ) from exc

        logger.info("=" * 60)
        logger.info("SUPERVISED FRAUD ESTIMATOR (LightGBM)")
        logger.info("=" * 60)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray(y).astype(int)
        self.feature_names_ = feature_names

        fraud_rate = float(y.mean())
        scale_pos_weight = (1.0 - fraud_rate) / max(fraud_rate, 1e-6)
        if self.class_weight != "balanced":
            scale_pos_weight = 1.0

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "verbosity": -1,
            "seed": self.random_state,
            "feature_pre_filter": False,
        }

        train_set = lgb.Dataset(X, label=y, feature_name=feature_names)
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=0)]

        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(
                np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0),
                label=np.asarray(y_val).astype(int),
                feature_name=feature_names,
                reference=train_set,
            )
            valid_sets.append(val_set)
            valid_names.append("valid")
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds,
                    verbose=False,
                )
            )

        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.is_fitted = True

        train_prob = self.predict_proba(X)
        from sklearn.metrics import roc_auc_score

        metrics = {"train_auc": float(roc_auc_score(y, train_prob))}
        if X_val is not None and y_val is not None:
            val_prob = self.predict_proba(X_val)
            metrics["valid_auc"] = float(roc_auc_score(y_val, val_prob))

        logger.info(f"  Train AUC: {metrics['train_auc']:.4f}")
        if "valid_auc" in metrics:
            logger.info(f"  Valid AUC: {metrics['valid_auc']:.4f}")
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(fraud) for each row."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("FraudEstimator has not been fitted.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X).astype(np.float32)

    def augment_features(
        self,
        X: np.ndarray,
        fraud_prob: np.ndarray,
        risk_scores: np.ndarray,
        cluster_features: np.ndarray,
    ) -> np.ndarray:
        """Stack base features with signals for RL / secondary models."""
        return np.hstack([
            X,
            fraud_prob.reshape(-1, 1),
            risk_scores.reshape(-1, 1),
            cluster_features,
        ]).astype(np.float32)

    @staticmethod
    def cluster_feature_matrix(cluster_results: Dict[str, np.ndarray]) -> np.ndarray:
        """Build numeric cluster feature block for RL state."""
        label = cluster_results["cluster_label"].astype(np.float32)
        label_norm = np.where(label >= 0, label / (label.max() + 1e-8), -1.0)
        return np.column_stack([
            label_norm,
            cluster_results["cluster_probability"].astype(np.float32),
            cluster_results["outlier_score"].astype(np.float32),
            cluster_results["centroid_distance"].astype(np.float32),
            cluster_results["cluster_fraud_rate"].astype(np.float32),
        ])

    def save(self, filepath: str) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted estimator.")
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_names": self.feature_names_},
            filepath,
        )
        logger.info(f"  Fraud estimator saved to: {filepath}")

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names_ = data.get("feature_names")
        self.is_fitted = True
        logger.info(f"  Fraud estimator loaded from: {filepath}")
