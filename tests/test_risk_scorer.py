"""Tests for risk scoring and anomaly detection."""

import numpy as np
import pandas as pd
import pytest

from src.anomaly.isolation_forest import AnomalyDetector
from src.anomaly.risk_scorer import RiskScorer


class TestAnomalyDetector:
    """Tests for Isolation Forest anomaly detection."""

    @pytest.fixture
    def sample_data(self):
        """Generate data with clear anomalies."""
        np.random.seed(42)
        normal = np.random.normal(0, 1, (950, 10))
        anomalies = np.random.normal(10, 2, (50, 10))
        return np.vstack([normal, anomalies])

    def test_fit_predict_returns_keys(self, sample_data):
        """Should return all expected keys."""
        detector = AnomalyDetector(contamination=0.05)
        results = detector.fit_predict(sample_data)

        assert "anomaly_label" in results
        assert "anomaly_score" in results
        assert "anomaly_score_normalized" in results

    def test_anomaly_label_values(self, sample_data):
        """Labels should be either 1 (normal) or -1 (anomaly)."""
        detector = AnomalyDetector(contamination=0.05)
        results = detector.fit_predict(sample_data)

        unique_labels = set(results["anomaly_label"])
        assert unique_labels.issubset({1, -1})

    def test_normalized_score_range(self, sample_data):
        """Normalized scores should be in [0, 1]."""
        detector = AnomalyDetector(contamination=0.05)
        results = detector.fit_predict(sample_data)

        assert results["anomaly_score_normalized"].min() >= 0.0
        assert results["anomaly_score_normalized"].max() <= 1.0

    def test_detects_anomalies(self, sample_data):
        """Should detect some anomalies."""
        detector = AnomalyDetector(contamination=0.05)
        results = detector.fit_predict(sample_data)

        n_anomalies = (results["anomaly_label"] == -1).sum()
        assert n_anomalies > 0, "No anomalies detected"

    def test_handles_nan_input(self):
        """Should handle NaN/inf in input."""
        data = np.random.normal(0, 1, (100, 5))
        data[10, 2] = np.nan
        data[20, 3] = np.inf

        detector = AnomalyDetector(contamination=0.05)
        results = detector.fit_predict(data)

        assert len(results["anomaly_label"]) == 100

    def test_predict_requires_fit(self):
        """predict() should raise if not fitted."""
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError):
            detector.predict(np.random.normal(0, 1, (10, 5)))


class TestRiskScorer:
    """Tests for composite risk scoring."""

    @pytest.fixture
    def risk_inputs(self):
        """Generate sample risk signal inputs."""
        np.random.seed(42)
        n = 200
        return {
            "anomaly_score": np.random.uniform(0, 1, n),
            "outlier_score": np.random.uniform(0, 1, n),
            "cluster_probability": np.random.uniform(0, 1, n),
            "centroid_distance": np.random.exponential(2, n),
            "cluster_fraud_rate": np.random.uniform(0, 0.1, n),
            "amount_zscore": np.random.normal(0, 2, n),
            "labels": np.random.choice([0, 1], n, p=[0.965, 0.035]),
        }

    def test_risk_score_range(self, risk_inputs):
        """Risk scores should be in [0, 1]."""
        scorer = RiskScorer()
        results = scorer.compute_risk_score(
            anomaly_score=risk_inputs["anomaly_score"],
            outlier_score=risk_inputs["outlier_score"],
            cluster_probability=risk_inputs["cluster_probability"],
            centroid_distance=risk_inputs["centroid_distance"],
            cluster_fraud_rate=risk_inputs["cluster_fraud_rate"],
            amount_zscore=risk_inputs["amount_zscore"],
        )

        assert results["risk_score"].min() >= 0.0
        assert results["risk_score"].max() <= 1.0

    def test_risk_categories(self, risk_inputs):
        """Risk categories should be valid strings."""
        scorer = RiskScorer()
        results = scorer.compute_risk_score(
            anomaly_score=risk_inputs["anomaly_score"],
            outlier_score=risk_inputs["outlier_score"],
            cluster_probability=risk_inputs["cluster_probability"],
            centroid_distance=risk_inputs["centroid_distance"],
            cluster_fraud_rate=risk_inputs["cluster_fraud_rate"],
        )

        valid_categories = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        actual_categories = set(results["risk_category"])
        assert actual_categories.issubset(valid_categories)

    def test_signals_dataframe(self, risk_inputs):
        """risk_signals should be a DataFrame with correct shape."""
        scorer = RiskScorer()
        results = scorer.compute_risk_score(
            anomaly_score=risk_inputs["anomaly_score"],
            outlier_score=risk_inputs["outlier_score"],
            cluster_probability=risk_inputs["cluster_probability"],
            centroid_distance=risk_inputs["centroid_distance"],
            cluster_fraud_rate=risk_inputs["cluster_fraud_rate"],
        )

        assert isinstance(results["risk_signals"], pd.DataFrame)
        assert len(results["risk_signals"]) == 200

    def test_calibration(self, risk_inputs):
        """Calibration should not crash and should update is_calibrated."""
        scorer = RiskScorer()
        results = scorer.compute_risk_score(
            anomaly_score=risk_inputs["anomaly_score"],
            outlier_score=risk_inputs["outlier_score"],
            cluster_probability=risk_inputs["cluster_probability"],
            centroid_distance=risk_inputs["centroid_distance"],
            cluster_fraud_rate=risk_inputs["cluster_fraud_rate"],
        )

        scorer.calibrate(results["risk_signals"], risk_inputs["labels"])
        assert scorer.is_calibrated is True
