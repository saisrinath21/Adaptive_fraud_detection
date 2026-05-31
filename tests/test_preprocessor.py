"""Tests for data preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import Preprocessor
from src.data.loader import DataLoader


class TestDataLoader:
    """Tests for DataLoader synthetic data generation."""

    def test_synthetic_data_shape(self):
        """Synthetic data should have expected shape and columns."""
        loader = DataLoader("fake/path.csv", "fake/path.csv")
        df = loader.load(sample_frac=0.1)

        assert len(df) > 0
        assert "TransactionID" in df.columns
        assert "isFraud" in df.columns
        assert "TransactionAmt" in df.columns
        assert "TransactionDT" in df.columns

    def test_synthetic_fraud_rate(self):
        """Synthetic data should have ~3.5% fraud rate."""
        loader = DataLoader("fake/path.csv", "fake/path.csv")
        df = loader.load()

        fraud_rate = df["isFraud"].mean()
        assert 0.02 <= fraud_rate <= 0.05, f"Unexpected fraud rate: {fraud_rate}"

    def test_synthetic_data_types(self):
        """Synthetic data should have both numeric and categorical columns."""
        loader = DataLoader("fake/path.csv", "fake/path.csv")
        df = loader.load(sample_frac=0.1)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 10, "Expected many numeric columns"

    def test_synthetic_has_missing_values(self):
        """Synthetic data should contain some missing values (realistic)."""
        loader = DataLoader("fake/path.csv", "fake/path.csv")
        df = loader.load(sample_frac=0.1)

        total_missing = df.isnull().sum().sum()
        assert total_missing > 0, "Synthetic data should have missing values"


class TestPreprocessor:
    """Tests for Preprocessor."""

    @pytest.fixture
    def sample_data(self):
        """Create a small sample dataset for testing."""
        np.random.seed(42)
        n = 500
        return pd.DataFrame({
            "TransactionID": range(n),
            "TransactionDT": np.random.randint(0, 15_000_000, n),
            "TransactionAmt": np.abs(np.random.lognormal(3, 1, n)),
            "isFraud": np.random.choice([0, 1], n, p=[0.965, 0.035]),
            "ProductCD": np.random.choice(["W", "C", "H"], n),
            "card1": np.random.randint(1000, 5000, n),
            "card4": np.random.choice(["visa", "mastercard"], n),
            "P_emaildomain": np.random.choice(
                ["gmail.com", "yahoo.com", np.nan], n
            ),
            "numeric_with_nan": np.where(
                np.random.random(n) > 0.3,
                np.random.normal(0, 1, n),
                np.nan,
            ),
            "mostly_missing": np.where(
                np.random.random(n) > 0.8,
                np.random.normal(0, 1, n),
                np.nan,
            ),
        })

    def test_fit_transform_no_nans(self, sample_data):
        """After preprocessing, no NaN values should remain in numeric cols."""
        preprocessor = Preprocessor(missing_threshold=0.7)
        result = preprocessor.fit_transform(sample_data)

        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not result[col].isnull().any(), f"NaN found in {col}"

    def test_high_missing_columns_dropped(self, sample_data):
        """Columns with >70% missing values should be dropped."""
        preprocessor = Preprocessor(missing_threshold=0.7)
        result = preprocessor.fit_transform(sample_data)

        assert "mostly_missing" not in result.columns

    def test_temporal_features_created(self, sample_data):
        """Temporal cyclical features should be created."""
        preprocessor = Preprocessor()
        result = preprocessor.fit_transform(sample_data)

        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "day_sin" in result.columns
        assert "day_cos" in result.columns

    def test_target_preserved(self, sample_data):
        """Target column should survive preprocessing."""
        preprocessor = Preprocessor()
        result = preprocessor.fit_transform(sample_data)

        assert "isFraud" in result.columns

    def test_split_data_stratified(self, sample_data):
        """Train/test split should maintain fraud rate."""
        preprocessor = Preprocessor()
        result = preprocessor.fit_transform(sample_data)
        X_train, X_test, y_train, y_test = preprocessor.split_data(result)

        train_fraud_rate = y_train.mean()
        test_fraud_rate = y_test.mean()

        assert abs(train_fraud_rate - test_fraud_rate) < 0.02

    def test_transform_consistency(self, sample_data):
        """transform() should produce same shape as fit_transform() (minus target)."""
        preprocessor = Preprocessor()
        fitted_result = preprocessor.fit_transform(sample_data)

        # Transform same data (simulating new data with same structure)
        transformed = preprocessor.transform(sample_data)

        # Should have similar number of columns
        assert abs(len(fitted_result.columns) - len(transformed.columns)) <= 2
