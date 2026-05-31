"""
Data Preprocessor Module
========================
Handles missing value imputation, categorical encoding, feature scaling,
and temporal feature extraction for the fraud detection pipeline.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Preprocess raw transaction data for downstream modeling.

    Handles:
    - Missing value imputation (median for numeric, mode for categorical)
    - Dropping high-missing-rate columns
    - Categorical encoding (label + target encoding)
    - Feature scaling (StandardScaler)
    - Temporal feature extraction from TransactionDT

    Parameters
    ----------
    missing_threshold : float
        Drop columns with more than this fraction of missing values.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(self, missing_threshold: float = 0.7, random_state: int = 42):
        self.missing_threshold = missing_threshold
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoding_maps: Dict[str, Dict] = {}
        self.numeric_medians: Dict[str, float] = {}
        self.categorical_modes: Dict[str, str] = {}
        self.dropped_columns: List[str] = []
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.is_fitted = False

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "isFraud"
    ) -> pd.DataFrame:
        """
        Fit preprocessor on training data and transform it.

        Parameters
        ----------
        df : pd.DataFrame
            Raw merged dataset.
        target_col : str
            Name of the target column.

        Returns
        -------
        pd.DataFrame
            Preprocessed dataset ready for feature engineering.
        """
        logger.info("=" * 60)
        logger.info("PREPROCESSING PIPELINE")
        logger.info("=" * 60)

        df = df.copy()

        # Step 1: Drop high-missing columns
        df = self._drop_high_missing(df, target_col)

        # Step 2: Identify feature types
        self._identify_features(df, target_col)

        # Step 3: Extract temporal features
        df = self._extract_temporal_features(df)

        # Step 4: Impute missing values
        df = self._fit_impute_missing(df)

        # Step 5: Encode categorical features
        df = self._fit_encode_categoricals(df, target_col)

        # Step 6: Scale numeric features
        df = self._fit_scale_numerics(df)

        self.is_fitted = True
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.

        Parameters
        ----------
        df : pd.DataFrame
            New raw data to transform.

        Returns
        -------
        pd.DataFrame
            Preprocessed dataset.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit_transform() first.")

        df = df.copy()

        # Drop same columns
        df = df.drop(columns=[c for c in self.dropped_columns if c in df.columns], errors="ignore")

        # Temporal features
        df = self._extract_temporal_features(df)

        # Impute
        df = self._transform_impute_missing(df)

        # Encode
        df = self._transform_encode_categoricals(df)

        # Scale
        df = self._transform_scale_numerics(df)

        return df

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _drop_high_missing(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Drop columns with missing fraction > threshold."""
        missing_frac = df.isnull().mean()
        self.dropped_columns = missing_frac[
            (missing_frac > self.missing_threshold) & (missing_frac.index != target_col)
        ].index.tolist()

        if self.dropped_columns:
            logger.info(
                f"Dropping {len(self.dropped_columns)} columns with >{self.missing_threshold:.0%} missing values"
            )
            df = df.drop(columns=self.dropped_columns)

        return df

    def _identify_features(self, df: pd.DataFrame, target_col: str) -> None:
        """Separate numeric and categorical features."""
        exclude_cols = {target_col, "TransactionID", "TransactionDT"}

        self.numeric_features = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]
        self.categorical_features = [
            col for col in df.select_dtypes(include=["object", "category"]).columns
            if col not in exclude_cols
        ]

        logger.info(f"  Numeric features: {len(self.numeric_features)}")
        logger.info(f"  Categorical features: {len(self.categorical_features)}")

    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert TransactionDT timedelta to cyclical time features."""
        if "TransactionDT" not in df.columns:
            return df

        # TransactionDT is seconds from a reference datetime
        # Extract hour-of-day and day-of-week using modular arithmetic
        seconds_in_day = 86400
        seconds_in_week = 604800

        df["hour_of_day"] = (df["TransactionDT"] % seconds_in_day) / 3600
        df["day_of_week"] = (df["TransactionDT"] % seconds_in_week) / seconds_in_day

        # Cyclical encoding (sin/cos) to preserve periodicity
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Drop raw temporal columns (keep cyclical)
        df = df.drop(columns=["hour_of_day", "day_of_week"], errors="ignore")

        # Add new numeric features to tracking
        temporal_features = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
        for feat in temporal_features:
            if feat not in self.numeric_features:
                self.numeric_features.append(feat)

        logger.info("  Extracted cyclical temporal features (hour_sin/cos, day_sin/cos)")

        return df

    def _fit_impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and apply missing value imputation."""
        # Numeric: median imputation
        for col in self.numeric_features:
            if col in df.columns and df[col].isnull().any():
                self.numeric_medians[col] = df[col].median()
                df[col] = df[col].fillna(self.numeric_medians[col])

        # Categorical: mode imputation with "Unknown" fallback
        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()
                self.categorical_modes[col] = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
                df[col] = df[col].fillna(self.categorical_modes[col])

        n_imputed = len(self.numeric_medians) + len(self.categorical_modes)
        logger.info(f"  Imputed missing values in {n_imputed} columns")

        return df

    def _transform_impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation to new data."""
        for col, median_val in self.numeric_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)

        for col, mode_val in self.categorical_modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val)

        # Fill any remaining NaNs
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)

        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna("Unknown")

        return df

    def _fit_encode_categoricals(
        self, df: pd.DataFrame, target_col: str
    ) -> pd.DataFrame:
        """Fit and apply categorical encoding."""
        # High-cardinality: target encoding
        high_card_cols = [
            col for col in self.categorical_features
            if col in df.columns and df[col].nunique() > 10
        ]
        # Low-cardinality: label encoding
        low_card_cols = [
            col for col in self.categorical_features
            if col in df.columns and df[col].nunique() <= 10
        ]

        # Target encoding for high-cardinality
        for col in high_card_cols:
            if target_col in df.columns:
                mapping = df.groupby(col)[target_col].mean().to_dict()
                global_mean = df[target_col].mean()
                self.target_encoding_maps[col] = {"mapping": mapping, "default": global_mean}
                df[col] = df[col].map(mapping).fillna(global_mean)
            else:
                # If no target available, use frequency encoding
                mapping = df[col].value_counts(normalize=True).to_dict()
                self.target_encoding_maps[col] = {"mapping": mapping, "default": 0.0}
                df[col] = df[col].map(mapping).fillna(0.0)

        # Label encoding for low-cardinality
        for col in low_card_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        logger.info(
            f"  Encoded {len(high_card_cols)} high-cardinality (target encoding), "
            f"{len(low_card_cols)} low-cardinality (label encoding)"
        )

        return df

    def _transform_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoding to new data."""
        for col, enc_info in self.target_encoding_maps.items():
            if col in df.columns:
                df[col] = df[col].map(enc_info["mapping"]).fillna(enc_info["default"])

        for col, le in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen labels gracefully
                known_classes = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in known_classes else -1
                )

        return df

    def _fit_scale_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and apply StandardScaler to numeric features."""
        # Get all current numeric columns (excluding target and ID)
        scale_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in {"isFraud", "TransactionID", "TransactionDT"}
        ]

        if scale_cols:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
            self.numeric_features = scale_cols  # Update with final list
            logger.info(f"  Scaled {len(scale_cols)} numeric features (StandardScaler)")

        return df

    def _transform_scale_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to new data."""
        scale_cols = [col for col in self.numeric_features if col in df.columns]
        if scale_cols:
            df[scale_cols] = self.scaler.transform(df[scale_cols])
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str = "isFraud",
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split preprocessed data into train/test sets with stratification.

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        exclude_cols = {target_col, "TransactionID", "TransactionDT"}
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        logger.info(
            f"  Train/test split: {len(X_train)} train, {len(X_test)} test "
            f"(fraud rate: train={y_train.mean():.3%}, test={y_test.mean():.3%})"
        )

        return X_train, X_test, y_train, y_test
