"""
Data Loader Module
==================
Handles loading and merging of the IEEE-CIS Fraud Detection dataset.
Supports both the real Kaggle dataset and synthetic data generation
for development/testing when the real dataset is unavailable.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Load and merge IEEE-CIS Fraud Detection data.

    Supports two modes:
    1. Real data: Load train_transaction.csv and train_identity.csv from disk
    2. Synthetic data: Generate realistic synthetic fraud data for development

    Parameters
    ----------
    transaction_path : str
        Path to train_transaction.csv.
    identity_path : str
        Path to train_identity.csv.
    """

    def __init__(self, transaction_path: str, identity_path: str):
        self.transaction_path = transaction_path
        self.identity_path = identity_path

    def load(self, sample_frac: Optional[float] = None) -> pd.DataFrame:
        """
        Load and merge transaction and identity datasets.

        Parameters
        ----------
        sample_frac : float, optional
            Fraction of data to sample (for faster development). If None, load all.

        Returns
        -------
        pd.DataFrame
            Merged dataset with optimized dtypes.
        """
        if os.path.exists(self.transaction_path) and os.path.exists(self.identity_path):
            return self._load_real_data(sample_frac)
        else:
            logger.warning(
                "IEEE-CIS dataset not found. Generating synthetic data for development."
            )
            return self._generate_synthetic_data(n_samples=50000 if sample_frac is None else int(50000 * sample_frac))

    def _load_real_data(self, sample_frac: Optional[float] = None) -> pd.DataFrame:
        """Load real IEEE-CIS data from CSV files."""
        logger.info(f"Loading transactions from: {self.transaction_path}")
        df_transaction = pd.read_csv(self.transaction_path)
        logger.info(f"  Transactions shape: {df_transaction.shape}")

        logger.info(f"Loading identity from: {self.identity_path}")
        df_identity = pd.read_csv(self.identity_path)
        logger.info(f"  Identity shape: {df_identity.shape}")

        # Merge on TransactionID (left join: not all transactions have identity info)
        logger.info("Merging datasets on TransactionID...")
        df = df_transaction.merge(df_identity, on="TransactionID", how="left")
        logger.info(f"  Merged shape: {df.shape}")

        # Optional sampling
        if sample_frac is not None and 0 < sample_frac < 1:
            df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
            logger.info(f"  Sampled to {len(df)} rows ({sample_frac:.0%})")

        # Optimize memory usage
        df = self._optimize_dtypes(df)

        return df

    def _generate_synthetic_data(self, n_samples: int = 50000) -> pd.DataFrame:
        """
        Generate synthetic fraud detection data mimicking IEEE-CIS structure.

        Creates realistic patterns where fraud transactions have distinct
        behavioral characteristics from legitimate transactions.
        """
        logger.info(f"Generating {n_samples} synthetic transactions...")
        np.random.seed(42)

        fraud_rate = 0.035
        n_fraud = int(n_samples * fraud_rate)
        n_legit = n_samples - n_fraud

        # --- Legitimate transactions ---
        legit = {
            "TransactionID": np.arange(n_legit),
            "isFraud": np.zeros(n_legit, dtype=int),
            "TransactionDT": np.sort(np.random.randint(0, 15_811_131, n_legit)),
            "TransactionAmt": np.abs(np.random.lognormal(3.5, 1.2, n_legit)),
            "ProductCD": np.random.choice(["W", "C", "H", "S", "R"], n_legit, p=[0.55, 0.15, 0.15, 0.10, 0.05]),
            "card1": np.random.randint(1000, 18000, n_legit),
            "card2": np.random.choice([100, 150, 200, 225, 300, 500], n_legit),
            "card3": np.random.choice([150, 185, 200], n_legit, p=[0.85, 0.10, 0.05]),
            "card4": np.random.choice(["visa", "mastercard", "discover", "american express"], n_legit, p=[0.55, 0.30, 0.10, 0.05]),
            "card5": np.random.choice([100, 117, 166, 200, 224, 226], n_legit),
            "card6": np.random.choice(["debit", "credit", "debit or credit", "charge card"], n_legit, p=[0.55, 0.35, 0.08, 0.02]),
            "addr1": np.random.randint(100, 500, n_legit).astype(float),
            "addr2": np.random.choice([87.0, 60.0, 96.0, 32.0], n_legit, p=[0.90, 0.05, 0.03, 0.02]),
            "dist1": np.abs(np.random.exponential(10, n_legit)),
            "P_emaildomain": np.random.choice(
                ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "anonymous.com", np.nan],
                n_legit, p=[0.35, 0.20, 0.15, 0.10, 0.05, 0.05, 0.10],
            ),
            "R_emaildomain": np.random.choice(
                ["gmail.com", "yahoo.com", "hotmail.com", np.nan],
                n_legit, p=[0.15, 0.05, 0.05, 0.75],
            ),
        }

        # --- Fraudulent transactions (different distributions) ---
        fraud = {
            "TransactionID": np.arange(n_legit, n_samples),
            "isFraud": np.ones(n_fraud, dtype=int),
            "TransactionDT": np.sort(np.random.randint(0, 15_811_131, n_fraud)),
            "TransactionAmt": np.abs(np.random.lognormal(4.5, 1.8, n_fraud)),  # Higher amounts
            "ProductCD": np.random.choice(["W", "C", "H", "S", "R"], n_fraud, p=[0.30, 0.25, 0.25, 0.10, 0.10]),
            "card1": np.random.randint(1000, 18000, n_fraud),
            "card2": np.random.choice([100, 150, 200, 225, 300, 500], n_fraud),
            "card3": np.random.choice([150, 185, 200], n_fraud, p=[0.70, 0.15, 0.15]),
            "card4": np.random.choice(["visa", "mastercard", "discover", "american express"], n_fraud, p=[0.40, 0.35, 0.15, 0.10]),
            "card5": np.random.choice([100, 117, 166, 200, 224, 226], n_fraud),
            "card6": np.random.choice(["debit", "credit", "debit or credit", "charge card"], n_fraud, p=[0.40, 0.45, 0.10, 0.05]),
            "addr1": np.random.randint(100, 500, n_fraud).astype(float),
            "addr2": np.random.choice([87.0, 60.0, 96.0, 32.0], n_fraud, p=[0.60, 0.20, 0.10, 0.10]),
            "dist1": np.abs(np.random.exponential(50, n_fraud)),  # Larger distances
            "P_emaildomain": np.random.choice(
                ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "protonmail.com", "anonymous.com", np.nan],
                n_fraud, p=[0.15, 0.10, 0.05, 0.05, 0.20, 0.25, 0.20],
            ),
            "R_emaildomain": np.random.choice(
                ["gmail.com", "yahoo.com", "hotmail.com", np.nan],
                n_fraud, p=[0.10, 0.05, 0.05, 0.80],
            ),
        }

        # Add C-features (counting)
        for i in range(1, 15):
            legit[f"C{i}"] = np.random.poisson(1.5, n_legit).astype(float)
            fraud[f"C{i}"] = np.random.poisson(3.0, n_fraud).astype(float)  # Higher counts for fraud

        # Add D-features (timedelta)
        for i in range(1, 16):
            legit[f"D{i}"] = np.where(
                np.random.random(n_legit) > 0.3,
                np.abs(np.random.exponential(20, n_legit)),
                np.nan,
            )
            fraud[f"D{i}"] = np.where(
                np.random.random(n_fraud) > 0.5,
                np.abs(np.random.exponential(5, n_fraud)),  # Shorter timedeltas
                np.nan,
            )

        # Add M-features (match, categorical)
        for i in range(1, 10):
            legit[f"M{i}"] = np.random.choice(["T", "F", np.nan], n_legit, p=[0.6, 0.2, 0.2])
            fraud[f"M{i}"] = np.random.choice(["T", "F", np.nan], n_fraud, p=[0.3, 0.4, 0.3])

        # Add V-features (engineered, subset for synthetic)
        for i in range(1, 50):
            legit[f"V{i}"] = np.random.normal(0, 1, n_legit)
            fraud[f"V{i}"] = np.random.normal(0.5, 1.5, n_fraud)  # Shifted distribution

        # Add identity-like features
        for col_name in ["id_01", "id_02", "id_03", "id_04", "id_05", "id_06"]:
            legit[col_name] = np.where(
                np.random.random(n_legit) > 0.4,
                np.random.normal(0, 10, n_legit),
                np.nan,
            )
            fraud[col_name] = np.where(
                np.random.random(n_fraud) > 0.6,
                np.random.normal(-5, 15, n_fraud),
                np.nan,
            )

        legit["DeviceType"] = np.random.choice(
            ["desktop", "mobile", np.nan], n_legit, p=[0.50, 0.35, 0.15]
        )
        fraud["DeviceType"] = np.random.choice(
            ["desktop", "mobile", np.nan], n_fraud, p=[0.30, 0.40, 0.30]
        )

        legit["DeviceInfo"] = np.random.choice(
            ["Windows", "iOS Device", "MacOS", "Trident/7.0", np.nan],
            n_legit, p=[0.30, 0.25, 0.15, 0.10, 0.20],
        )
        fraud["DeviceInfo"] = np.random.choice(
            ["Windows", "iOS Device", "MacOS", "Linux", np.nan],
            n_fraud, p=[0.20, 0.10, 0.05, 0.25, 0.40],
        )

        # Combine and shuffle
        df_legit = pd.DataFrame(legit)
        df_fraud = pd.DataFrame(fraud)
        df = pd.concat([df_legit, df_fraud], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df["TransactionID"] = np.arange(len(df))

        logger.info(
            f"  Synthetic data generated: {len(df)} transactions, "
            f"{df['isFraud'].sum()} fraud ({df['isFraud'].mean():.1%})"
        )

        return df

    @staticmethod
    def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Downcast numeric types to reduce memory usage.

        Iterates columns individually to avoid pandas block-consolidation
        that can trigger large intermediate memory allocations.
        """
        initial_mem = df.memory_usage(deep=True).sum() / 1024**2

        for col in df.columns:
            col_dtype = df[col].dtype
            if col_dtype == "int64":
                df[col] = pd.to_numeric(df[col], downcast="integer")
            elif col_dtype == "float64":
                df[col] = pd.to_numeric(df[col], downcast="float")

        final_mem = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(
            f"  Memory optimized: {initial_mem:.1f}MB → {final_mem:.1f}MB "
            f"({(1 - final_mem / initial_mem) * 100:.0f}% reduction)"
        )

        return df

