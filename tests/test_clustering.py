"""Tests for behavioral clustering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.clustering.pca_reducer import PCAReducer
from src.clustering.behavior_aggregator import BehaviorAggregator
from src.clustering.hdbscan_clustering import BehavioralClusterer


class TestPCAReducer:
    """Tests for PCA embedding reducer."""

    def test_fit_transform_shape(self):
        np.random.seed(42)
        X = np.random.normal(0, 1, (200, 50)).astype(np.float32)
        reducer = PCAReducer(n_components=20)
        embedding = reducer.fit_transform(X)
        assert embedding.shape == (200, 20)

    def test_transform_after_fit(self):
        np.random.seed(42)
        X = np.random.normal(0, 1, (200, 30)).astype(np.float32)
        reducer = PCAReducer(n_components=10)
        reducer.fit_transform(X[:150])
        out = reducer.transform(X[150:])
        assert out.shape == (50, 10)

    def test_fit_transform_2d(self):
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 25)).astype(np.float32)
        embedding_2d = PCAReducer.fit_transform_2d(X)
        assert embedding_2d.shape == (100, 2)


class TestBehaviorAggregator:
    """Tests for entity-level behavior aggregation."""

    def test_build_behavior_id(self):
        df = pd.DataFrame({
            "card1": [1, 1, 2],
            "addr1": [10, 10, 20],
            "DeviceType": ["desktop", "desktop", "mobile"],
            "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com"],
        })
        agg = BehaviorAggregator()
        bids = agg.build_behavior_id(df)
        assert bids.nunique() == 2

    def test_aggregate_and_map(self):
        np.random.seed(0)
        n = 120
        df = pd.DataFrame({
            "card1": np.repeat([1, 2, 3], n // 3),
            "addr1": np.repeat([10, 20, 30], n // 3),
            "DeviceType": "desktop",
            "P_emaildomain": "gmail.com",
        })
        X = np.random.normal(0, 1, (n, 8)).astype(np.float32)
        y = (np.random.rand(n) > 0.9).astype(int)
        agg = BehaviorAggregator()
        bids = agg.build_behavior_id(df)
        entity_features, codes, _ = agg.aggregate_entities(X, bids, y=y)

        entity_results = {
            "cluster_label": np.array([0, 1, -1]),
            "cluster_probability": np.array([1.0, 0.8, 0.1]),
            "outlier_score": np.array([0.0, 0.2, 0.9]),
            "centroid_distance": np.array([0.1, 0.3, 1.0]),
            "cluster_fraud_rate": np.array([0.01, 0.05, 0.5]),
        }
        mapped = agg.map_entity_results_to_transactions(entity_results, codes)
        assert len(mapped["cluster_label"]) == n


class TestBehavioralClusterer:
    """Tests for HDBSCAN clustering."""

    @pytest.fixture
    def clusterable_data(self):
        """Generate data with clear cluster structure in low dimensions."""
        np.random.seed(42)
        cluster1 = np.random.normal([0, 0], 0.5, (150, 2))
        cluster2 = np.random.normal([5, 5], 0.5, (150, 2))
        cluster3 = np.random.normal([10, 0], 0.5, (150, 2))
        noise = np.random.uniform(-5, 15, (50, 2))

        data = np.vstack([cluster1, cluster2, cluster3, noise])
        labels = np.concatenate([
            np.zeros(150), np.zeros(150), np.zeros(150),
            np.ones(50),
        ]).astype(int)

        return data, labels

    def test_fit_predict_returns_all_keys(self, clusterable_data):
        data, labels = clusterable_data
        clusterer = BehavioralClusterer(min_cluster_size=20, min_samples=5)
        results = clusterer.fit_predict(data, labels=labels)

        expected_keys = {
            "cluster_label", "cluster_probability", "outlier_score",
            "centroid_distance", "cluster_fraud_rate",
        }
        assert expected_keys == set(results.keys())

    def test_cluster_label_shape(self, clusterable_data):
        data, labels = clusterable_data
        clusterer = BehavioralClusterer(min_cluster_size=20, min_samples=5)
        results = clusterer.fit_predict(data, labels=labels)
        assert len(results["cluster_label"]) == len(data)

    def test_identifies_clusters(self, clusterable_data):
        data, labels = clusterable_data
        clusterer = BehavioralClusterer(min_cluster_size=20, min_samples=5)
        results = clusterer.fit_predict(data, labels=labels)

        n_clusters = len(set(results["cluster_label"])) - (
            1 if -1 in results["cluster_label"] else 0
        )
        assert n_clusters >= 2, f"Only found {n_clusters} clusters"

    def test_outlier_scores_bounded(self, clusterable_data):
        data, labels = clusterable_data
        clusterer = BehavioralClusterer(min_cluster_size=20, min_samples=5)
        results = clusterer.fit_predict(data, labels=labels)
        assert (results["outlier_score"] >= 0).all()

    def test_probability_bounded(self, clusterable_data):
        data, labels = clusterable_data
        clusterer = BehavioralClusterer(min_cluster_size=20, min_samples=5)
        results = clusterer.fit_predict(data, labels=labels)
        assert (results["cluster_probability"] >= 0).all()
        assert (results["cluster_probability"] <= 1).all()

    def test_predict_requires_fit(self):
        clusterer = BehavioralClusterer()
        with pytest.raises(RuntimeError):
            clusterer.predict(np.random.normal(0, 1, (10, 5)))
