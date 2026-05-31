"""
PCA Embedding Reducer
=======================
Linear dimensionality reduction for entity-level behavioral vectors
before HDBSCAN clustering.
"""

import os
from typing import Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PCAReducer:
    """
    PCA-based reducer for behavioral embeddings.

    Parameters
    ----------
    n_components : int
        Number of principal components (default 20).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 20,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.pca: Optional[PCA] = None
        self.is_fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and return reduced embeddings."""
        logger.info("=" * 60)
        logger.info("PCA EMBEDDING REDUCTION")
        logger.info("=" * 60)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        if n_comp < 1:
            raise ValueError(f"Cannot run PCA on array with shape {X.shape}")

        logger.info(f"  Input shape: {X.shape}")
        logger.info(f"  n_components: {n_comp}")

        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        embedding = self.pca.fit_transform(X).astype(np.float32)
        self.is_fitted = True

        explained = float(self.pca.explained_variance_ratio_.sum())
        logger.info(f"  Output shape: {embedding.shape}")
        logger.info(f"  Explained variance ratio: {explained:.1%}")
        return embedding

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data with fitted PCA."""
        if not self.is_fitted or self.pca is None:
            raise RuntimeError("PCAReducer has not been fitted. Call fit_transform() first.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return self.pca.transform(X).astype(np.float32)

    @staticmethod
    def fit_transform_2d(X: np.ndarray, random_state: int = 42) -> np.ndarray:
        """One-off 2D PCA projection for cluster visualization plots."""
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        n_comp = min(2, X.shape[1], max(1, X.shape[0] - 1))
        pca = PCA(n_components=n_comp, random_state=random_state)
        return pca.fit_transform(X).astype(np.float32)

    def save(self, filepath: str) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted PCA reducer.")
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        joblib.dump({"pca": self.pca, "n_components": self.n_components}, filepath)
        logger.info(f"  PCA reducer saved to: {filepath}")

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.pca = data["pca"]
        self.n_components = data.get("n_components", self.pca.n_components_)
        self.is_fitted = True
        logger.info(f"  PCA reducer loaded from: {filepath}")
