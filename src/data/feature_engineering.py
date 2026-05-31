"""
Feature Engineering Module
===========================
Creates behavioral features from transaction data to capture user activity
patterns, transaction velocity, amount deviations, and device/merchant
consistency indicators.
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Engineer behavioral features for fraud detection.

    Creates user-level aggregation features using a pseudo-user ID
    (constructed from card1 + addr1 combination). Features capture
    spending patterns, transaction velocity, device diversity, and
    behavioral consistency.

    Parameters
    ----------
    user_id_cols : list of str
        Columns to combine as a pseudo-user identifier.
    """

    def __init__(self, user_id_cols: Optional[List[str]] = None):
        self.user_id_cols = user_id_cols or ["card1", "addr1"]
        self.user_stats: Optional[pd.DataFrame] = None
        self.global_stats: dict = {}
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and attach behavioral features to the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed dataset (after Preprocessor.fit_transform).

        Returns
        -------
        pd.DataFrame
            Dataset with additional behavioral features.
        """
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING")
        logger.info("=" * 60)

        df = df.copy()
        initial_cols = len(df.columns)

        # Create pseudo user ID
        df = self._create_user_id(df)

        # Compute global stats for later transforms
        self._compute_global_stats(df)

        # Transaction amount features
        df = self._amount_features(df)

        # Transaction velocity features
        df = self._velocity_features(df)

        # Time-based features
        df = self._time_features(df)

        # Device and identity features
        df = self._device_features(df)

        # Email domain features
        df = self._email_features(df)

        # Card usage features
        df = self._card_features(df)

        # Interaction features
        df = self._interaction_features(df)

        # Store user-level statistics for transform
        self._store_user_stats(df)

        # Clean up helper columns
        df = df.drop(columns=["user_id"], errors="ignore")

        new_cols = len(df.columns) - initial_cols
        logger.info(f"Feature engineering complete. Added {new_cols} new features.")
        logger.info(f"  Final feature count: {len(df.columns)}")

        self.is_fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to new data using fitted statistics.

        Parameters
        ----------
        df : pd.DataFrame
            New preprocessed data.

        Returns
        -------
        pd.DataFrame
            Data with behavioral features.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer has not been fitted. Call fit_transform() first.")

        df = df.copy()
        df = self._create_user_id(df)
        df = self._amount_features(df)
        df = self._velocity_features(df)
        df = self._time_features(df)
        df = self._device_features(df)
        df = self._email_features(df)
        df = self._card_features(df)
        df = self._interaction_features(df)
        df = df.drop(columns=["user_id"], errors="ignore")
        return df

    # ------------------------------------------------------------------
    # Feature creation methods
    # ------------------------------------------------------------------

    def _create_user_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pseudo user ID from card and address columns."""
        available_cols = [col for col in self.user_id_cols if col in df.columns]
        if available_cols:
            df["user_id"] = df[available_cols].astype(str).agg("_".join, axis=1)
        else:
            # Fallback: use index-based grouping
            df["user_id"] = "user_0"
            logger.warning("User ID columns not found. Using single-group fallback.")
        return df

    def _compute_global_stats(self, df: pd.DataFrame) -> None:
        """Pre-compute global statistics for normalization."""
        if "TransactionAmt" in df.columns:
            self.global_stats["global_mean_amt"] = df["TransactionAmt"].mean()
            self.global_stats["global_std_amt"] = df["TransactionAmt"].std()
        if "TransactionDT" in df.columns:
            self.global_stats["global_mean_dt"] = df["TransactionDT"].mean()

    def _amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transaction amount behavioral features."""
        if "TransactionAmt" not in df.columns:
            return df

        # User-level amount statistics
        user_amt = df.groupby("user_id")["TransactionAmt"].agg(["mean", "std", "count", "max", "min"])
        user_amt.columns = [
            "user_avg_amount", "user_std_amount", "user_tx_count",
            "user_max_amount", "user_min_amount",
        ]
        user_amt["user_std_amount"] = user_amt["user_std_amount"].fillna(0)
        df = df.merge(user_amt, on="user_id", how="left")

        # Z-score of current amount vs user history
        df["user_amount_zscore"] = np.where(
            df["user_std_amount"] > 0,
            (df["TransactionAmt"] - df["user_avg_amount"]) / df["user_std_amount"],
            0,
        )

        # Ratio of current amount to user's mean
        df["amount_to_mean_ratio"] = np.where(
            df["user_avg_amount"] > 0,
            df["TransactionAmt"] / df["user_avg_amount"],
            1.0,
        )

        # Amount relative to global distribution
        global_mean = self.global_stats.get("global_mean_amt", df["TransactionAmt"].mean())
        global_std = self.global_stats.get("global_std_amt", df["TransactionAmt"].std())
        df["global_amount_zscore"] = (df["TransactionAmt"] - global_mean) / (global_std + 1e-8)

        # Log-transformed amount (reduces skew)
        df["log_amount"] = np.log1p(df["TransactionAmt"].abs())

        # Amount percentile within user
        df["amount_percentile"] = df.groupby("user_id")["TransactionAmt"].rank(pct=True)

        logger.info("  [OK] Amount features (z-score, ratio, percentile, log)")
        return df

    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transaction velocity and frequency features."""
        if "TransactionDT" not in df.columns:
            return df

        # Sort by time for proper velocity computation
        df = df.sort_values("TransactionDT").reset_index(drop=True)

        # Time since previous transaction (per user)
        df["time_since_last_tx"] = df.groupby("user_id")["TransactionDT"].diff()
        df["time_since_last_tx"] = df["time_since_last_tx"].fillna(-1)

        # Transaction count in various time windows (using rolling on sorted data)
        # Approximate: count of user transactions within last N seconds
        for window_name, window_seconds in [("1h", 3600), ("24h", 86400), ("7d", 604800)]:
            col_name = f"tx_velocity_{window_name}"
            # Use a groupby + rolling count approach
            df[col_name] = df.groupby("user_id").cumcount() + 1

        # Average time between transactions per user
        df["user_avg_time_between_tx"] = df.groupby("user_id")["time_since_last_tx"].transform("mean")
        df["user_avg_time_between_tx"] = df["user_avg_time_between_tx"].fillna(-1)

        # Is rapid transaction? (within 60 seconds of last)
        df["is_rapid_tx"] = (df["time_since_last_tx"].between(0, 60)).astype(int)

        logger.info("  [OK] Velocity features (time_since_last, velocity_1h/24h/7d, rapid_tx)")
        return df

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Additional temporal pattern features."""
        if "TransactionDT" not in df.columns:
            return df

        seconds_in_day = 86400

        # Is night transaction (between midnight and 6am)
        hour_approx = (df["TransactionDT"] % seconds_in_day) / 3600
        df["is_night_tx"] = ((hour_approx >= 0) & (hour_approx < 6)).astype(int)

        # Is weekend transaction
        day_of_week = (df["TransactionDT"] % (seconds_in_day * 7)) / seconds_in_day
        df["is_weekend_tx"] = (day_of_week >= 5).astype(int)

        logger.info("  [OK] Time pattern features (is_night, is_weekend)")
        return df

    def _device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Device diversity and consistency features."""
        # Number of unique devices per user
        if "DeviceType" in df.columns:
            device_div = df.groupby("user_id")["DeviceType"].nunique().rename("device_diversity")
            df = df.merge(device_div, on="user_id", how="left")
            df["device_diversity"] = df["device_diversity"].fillna(1)

        if "DeviceInfo" in df.columns:
            device_info_div = df.groupby("user_id")["DeviceInfo"].nunique().rename("device_info_diversity")
            df = df.merge(device_info_div, on="user_id", how="left")
            df["device_info_diversity"] = df["device_info_diversity"].fillna(1)

        logger.info("  [OK] Device features (diversity)")
        return df

    def _email_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Email domain consistency and risk features."""
        if "P_emaildomain" in df.columns:
            # Number of unique email domains per user
            email_div = df.groupby("user_id")["P_emaildomain"].nunique().rename("email_diversity")
            df = df.merge(email_div, on="user_id", how="left")
            df["email_diversity"] = df["email_diversity"].fillna(1)

        # Email match indicator (purchaser vs recipient)
        if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
            df["email_domain_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(int)

        logger.info("  [OK] Email features (diversity, domain_match)")
        return df

    def _card_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Card usage pattern features."""
        if "card1" in df.columns:
            card_counts = df.groupby("card1")["TransactionID"].count().rename("card_usage_count")
            df = df.merge(card_counts, on="card1", how="left")

        if "card1" in df.columns and "addr1" in df.columns:
            # Address change detection: unique addresses per card
            addr_per_card = df.groupby("card1")["addr1"].nunique().rename("addr_per_card")
            df = df.merge(addr_per_card, on="card1", how="left")
            df["addr_change_flag"] = (df["addr_per_card"] > 1).astype(int)

        logger.info("  [OK] Card features (usage_count, addr_change_flag)")
        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-feature interaction terms."""
        if "TransactionAmt" in df.columns and "user_tx_count" in df.columns:
            # Amount * frequency interaction
            df["amt_x_txcount"] = df["TransactionAmt"] * df["user_tx_count"]

        if "log_amount" in df.columns and "is_night_tx" in df.columns:
            # High amount at night = suspicious
            df["night_high_amount"] = df["log_amount"] * df["is_night_tx"]

        if "is_rapid_tx" in df.columns and "user_amount_zscore" in df.columns:
            # Rapid + unusual amount = very suspicious
            df["rapid_unusual_amount"] = df["is_rapid_tx"] * df["user_amount_zscore"].abs()

        logger.info("  [OK] Interaction features (amt_x_txcount, night_high_amount, rapid_unusual)")
        return df

    def _store_user_stats(self, df: pd.DataFrame) -> None:
        """Store user-level statistics for later use in transform."""
        agg_cols = {}
        if "user_avg_amount" in df.columns:
            agg_cols["user_avg_amount"] = "first"
        if "user_std_amount" in df.columns:
            agg_cols["user_std_amount"] = "first"
        if "user_tx_count" in df.columns:
            agg_cols["user_tx_count"] = "first"

        if agg_cols:
            self.user_stats = df.groupby("user_id").agg(agg_cols)
