"""
HDBSCAN Behavioral Clustering Module
======================================
Applies HDBSCAN to PCA-reduced behavioral entity embeddings.
Outlier/noise points frequently correspond to fraudulent activity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import joblib
import os

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BehavioralClusterer:
    """
    HDBSCAN-based behavioral clustering for fraud detection.

    Identifies groups of transactions with similar behavioral patterns.
    Transactions that don't belong to any cluster (noise) are flagged as
    potential fraud. Outputs include cluster labels, membership probabilities,
    and outlier scores.

    Parameters
    ----------
    min_cluster_size : int
        Minimum cluster size for HDBSCAN.
    min_samples : int
        Minimum number of samples in neighborhood for core points.
    metric : str
        Distance metric.
    cluster_selection_method : str
        Method for selecting flat clusters from hierarchy ('eom' or 'leaf').
    """

    def __init__(
        self,
        min_cluster_size: int = 100,
        min_samples: int = 15,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.clusterer = None
        self.cluster_stats: Optional[pd.DataFrame] = None
        self.cluster_centroids: Optional[Dict[int, np.ndarray]] = None
        self.is_fitted = False

    def fit_predict(
        self, embedding: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Fit HDBSCAN and extract clustering features.

        Parameters
        ----------
        embedding : np.ndarray of shape (n_samples, n_components)
            PCA-reduced entity embeddings.
        labels : np.ndarray, optional
            True fraud labels (for computing cluster fraud rates).

        Returns
        -------
        dict
            Dictionary with keys:
            - 'cluster_label': int array, -1 = noise
            - 'cluster_probability': float array, membership confidence
            - 'outlier_score': float array, GLOSH outlier score
            - 'centroid_distance': float array, distance to cluster centroid
            - 'cluster_fraud_rate': float array, fraud rate of assigned cluster
        """
        import hdbscan

        logger.info("=" * 60)
        logger.info("HDBSCAN BEHAVIORAL CLUSTERING")
        logger.info("=" * 60)
        logger.info(f"  Input embedding shape: {embedding.shape}")
        logger.info(f"  min_cluster_size: {self.min_cluster_size}")
        logger.info(f"  min_samples: {self.min_samples}")

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            gen_min_span_tree=True,
            prediction_data=True,
        )

        cluster_labels = self.clusterer.fit_predict(embedding)
        self.is_fitted = True

        # Extract clustering features
        results = self._extract_features(embedding, cluster_labels, labels)

        # Log cluster summary
        self._log_summary(cluster_labels, labels)

        return results

    def predict(self, embedding: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict cluster membership for new data.

        Parameters
        ----------
        embedding : np.ndarray
            PCA-reduced new entity embeddings.

        Returns
        -------
        dict
            Clustering features for new data.
        """
        import hdbscan

        if not self.is_fitted:
            raise RuntimeError("Clusterer has not been fitted.")

        labels, probs = hdbscan.approximate_predict(self.clusterer, embedding)

        results = {
            "cluster_label": labels,
            "cluster_probability": probs,
            "outlier_score": np.zeros(len(labels)),  # Not available for new data
        }

        # Compute centroid distances
        results["centroid_distance"] = self._compute_centroid_distances(
            embedding, labels
        )

        # Map cluster fraud rates
        if self.cluster_stats is not None and "fraud_rate" in self.cluster_stats.columns:
            fraud_rate_map = self.cluster_stats["fraud_rate"].to_dict()
            noise_fraud_rate = fraud_rate_map.get(-1, 0.5)
            results["cluster_fraud_rate"] = np.array([
                fraud_rate_map.get(l, noise_fraud_rate) for l in labels
            ])
        else:
            results["cluster_fraud_rate"] = np.full(len(labels), 0.5)

        return results

    def _extract_features(
        self,
        embedding: np.ndarray,
        cluster_labels: np.ndarray,
        labels: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Extract all clustering-derived features."""
        results = {
            "cluster_label": cluster_labels,
            "cluster_probability": self.clusterer.probabilities_,
            # NaN = point not scored as outlier by GLOSH; treat as 0
            "outlier_score": np.nan_to_num(
                self.clusterer.outlier_scores_, nan=0.0
            ),
        }

        # Compute cluster centroids
        self._compute_centroids(embedding, cluster_labels)

        # Centroid distance for each point
        results["centroid_distance"] = self._compute_centroid_distances(
            embedding, cluster_labels
        )

        # Compute cluster-level statistics
        self._compute_cluster_stats(cluster_labels, labels)

        # Map cluster fraud rate to each transaction
        if self.cluster_stats is not None and "fraud_rate" in self.cluster_stats.columns:
            fraud_rate_map = self.cluster_stats["fraud_rate"].to_dict()
            noise_fraud_rate = fraud_rate_map.get(-1, 0.5)
            results["cluster_fraud_rate"] = np.array([
                fraud_rate_map.get(l, noise_fraud_rate) for l in cluster_labels
            ])
        else:
            results["cluster_fraud_rate"] = np.full(len(cluster_labels), 0.0)

        return results

    def _compute_centroids(
        self, embedding: np.ndarray, cluster_labels: np.ndarray
    ) -> None:
        """Compute centroid for each cluster."""
        self.cluster_centroids = {}
        unique_labels = set(cluster_labels)

        for label in unique_labels:
            if label == -1:
                continue
            mask = cluster_labels == label
            self.cluster_centroids[label] = embedding[mask].mean(axis=0)

    def _compute_centroid_distances(
        self, embedding: np.ndarray, cluster_labels: np.ndarray
    ) -> np.ndarray:
        """Compute distance from each point to its cluster centroid."""
        distances = np.zeros(len(embedding))

        if self.cluster_centroids is None:
            return distances

        for label, centroid in self.cluster_centroids.items():
            mask = cluster_labels == label
            if mask.any():
                diff = embedding[mask] - centroid
                distances[mask] = np.linalg.norm(diff, axis=1)

        # Noise points: use max distance as penalty
        noise_mask = cluster_labels == -1
        if noise_mask.any() and distances.max() > 0:
            distances[noise_mask] = distances[~noise_mask].max() * 1.5

        return distances

    def _compute_cluster_stats(
        self,
        cluster_labels: np.ndarray,
        labels: Optional[np.ndarray],
    ) -> None:
        """Compute per-cluster statistics."""
        stats = pd.DataFrame({"cluster": cluster_labels})

        if labels is not None:
            stats["is_fraud"] = labels.values if hasattr(labels, "values") else labels

        cluster_info = stats.groupby("cluster").agg(
            size=("cluster", "size"),
        )

        if labels is not None:
            fraud_rates = stats.groupby("cluster")["is_fraud"].mean()
            cluster_info["fraud_rate"] = fraud_rates

        self.cluster_stats = cluster_info
        logger.info(f"  Computed statistics for {len(cluster_info)} clusters")

    def _log_summary(
        self, cluster_labels: np.ndarray, labels: Optional[np.ndarray]
    ) -> None:
        """Log clustering summary statistics."""
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = (cluster_labels == -1).sum()
        noise_pct = n_noise / len(cluster_labels) * 100

        logger.info(f"  Clusters found: {n_clusters}")
        logger.info(f"  Noise points: {n_noise} ({noise_pct:.1f}%)")

        if labels is not None:
            fraud_labels = labels.values if hasattr(labels, "values") else labels
            noise_fraud = fraud_labels[cluster_labels == -1].mean() if n_noise > 0 else 0
            cluster_fraud = fraud_labels[cluster_labels != -1].mean() if (cluster_labels != -1).any() else 0
            logger.info(f"  Fraud rate in noise: {noise_fraud:.1%}")
            logger.info(f"  Fraud rate in clusters: {cluster_fraud:.1%}")

        if self.cluster_stats is not None:
            logger.info("\n  Cluster Summary:")
            logger.info(f"  {'Cluster':>8} {'Size':>8} {'Fraud%':>8}")
            logger.info(f"  {'-' * 26}")
            for idx, row in self.cluster_stats.iterrows():
                fraud_str = f"{row.get('fraud_rate', 0):.1%}" if "fraud_rate" in row else "N/A"
                logger.info(f"  {idx:>8} {int(row['size']):>8} {fraud_str:>8}")

    def save(self, filepath: str) -> None:
        """Save fitted clusterer to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_data = {
            "clusterer": self.clusterer,
            "cluster_stats": self.cluster_stats,
            "cluster_centroids": self.cluster_centroids,
        }
        joblib.dump(save_data, filepath)
        logger.info(f"  Clusterer saved to: {filepath}")

    def load(self, filepath: str) -> None:
        """Load a previously fitted clusterer."""
        save_data = joblib.load(filepath)
        self.clusterer = save_data["clusterer"]
        self.cluster_stats = save_data["cluster_stats"]
        self.cluster_centroids = save_data["cluster_centroids"]
        self.is_fitted = True
        logger.info(f"  Clusterer loaded from: {filepath}")
