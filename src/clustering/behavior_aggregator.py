"""
Behavior Aggregation Module
===========================
Groups transactions by behavioral entity (card + address + device + email)
and builds entity-level feature vectors for PCA + HDBSCAN clustering.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BehaviorAggregator:
    """
    Build behavioral entity IDs and aggregate transaction features per entity.

    Parameters
    ----------
    entity_id_cols : list of str
        Columns combined to form ``behavior_id`` (e.g. card1, addr1, DeviceType).
    """

    def __init__(
        self,
        entity_id_cols: Optional[List[str]] = None,
    ):
        self.entity_id_cols = entity_id_cols or [
            "card1",
            "addr1",
            "DeviceType",
            "P_emaildomain",
        ]
        self.entity_index_: Optional[pd.Index] = None
        self.entity_id_to_idx_: Optional[Dict[str, int]] = None

    def build_behavior_id(self, df: pd.DataFrame) -> pd.Series:
        """Create a stable behavioral entity identifier per row."""
        available = [c for c in self.entity_id_cols if c in df.columns]
        if not available:
            logger.warning(
                f"  No entity ID columns found among {self.entity_id_cols}. "
                "Using row index as behavior_id."
            )
            return pd.Series(df.index.astype(str), index=df.index, name="behavior_id")

        behavior_id = (
            df[available].astype(str).fillna("NA").agg("|".join, axis=1)
        )
        behavior_id.name = "behavior_id"
        logger.info(
            f"  Built behavior_id from {available} "
            f"({behavior_id.nunique():,} unique entities / {len(behavior_id):,} rows)"
        )
        return behavior_id

    def aggregate_entities(
        self,
        X: np.ndarray,
        behavior_ids: pd.Series,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate transaction features to entity level.

        Returns
        -------
        entity_features : np.ndarray of shape (n_entities, n_features * 2 + 1)
            Per-entity mean, std, and log transaction count.
        entity_codes : np.ndarray of int
            Entity index per transaction (for mapping back).
        entity_fraud_rate : np.ndarray of shape (n_entities,)
            Mean fraud label per entity (zeros if labels absent).
        """
        logger.info("=" * 60)
        logger.info("BEHAVIOR AGGREGATION (entity-level)")
        logger.info("=" * 60)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        ids = behavior_ids.astype(str).values

        feat_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        feat_df["behavior_id"] = ids
        if y is not None:
            feat_df["is_fraud"] = np.asarray(y).astype(np.float64)

        grouped = feat_df.groupby("behavior_id", sort=False)
        entity_mean = grouped.mean(numeric_only=True)
        if "is_fraud" in entity_mean.columns:
            entity_mean = entity_mean.drop(columns=["is_fraud"])

        entity_std = grouped.std(numeric_only=True).fillna(0.0)
        if "is_fraud" in entity_std.columns:
            entity_std = entity_std.drop(columns=["is_fraud"])

        entity_count = grouped.size().rename("tx_count")

        entity_features = np.hstack([
            entity_mean.values,
            entity_std.values,
            np.log1p(entity_count.values).reshape(-1, 1),
        ]).astype(np.float32)

        if y is not None:
            entity_fraud_rate = (
                feat_df.groupby("behavior_id")["is_fraud"].mean().values.astype(np.float32)
            )
        else:
            entity_fraud_rate = np.zeros(len(entity_mean), dtype=np.float32)

        self.entity_index_ = entity_mean.index
        self.entity_id_to_idx_ = {
            bid: i for i, bid in enumerate(self.entity_index_)
        }
        entity_codes = np.array(
            [self.entity_id_to_idx_.get(bid, -1) for bid in ids], dtype=np.int32
        )

        logger.info(
            f"  Aggregated {len(ids):,} transactions -> "
            f"{len(self.entity_index_):,} entities, "
            f"feature dim={entity_features.shape[1]}"
        )
        return entity_features, entity_codes, entity_fraud_rate

    def entity_codes_for_ids(self, behavior_ids: pd.Series) -> np.ndarray:
        """Map behavior IDs to entity indices (unknown entities -> -1)."""
        if self.entity_id_to_idx_ is None:
            raise RuntimeError("Call aggregate_entities() before entity_codes_for_ids().")
        ids = behavior_ids.astype(str).values
        return np.array(
            [self.entity_id_to_idx_.get(bid, -1) for bid in ids], dtype=np.int32
        )

    def map_entity_results_to_transactions(
        self,
        entity_results: Dict[str, np.ndarray],
        entity_codes: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Broadcast per-entity clustering outputs to each transaction."""
        n_tx = len(entity_codes)
        mapped: Dict[str, np.ndarray] = {}
        for key, values in entity_results.items():
            arr = np.asarray(values)
            if len(arr) != len(self.entity_index_):
                raise ValueError(
                    f"Entity result length {len(arr)} != n_entities {len(self.entity_index_)}"
                )
            safe_codes = np.clip(entity_codes, 0, len(arr) - 1)
            out = arr[safe_codes].copy()
            out[entity_codes < 0] = arr.max() if key == "centroid_distance" else (
                -1 if key == "cluster_label" else 0.5
            )
            if key == "cluster_label":
                out[entity_codes < 0] = -1
            elif key == "cluster_fraud_rate":
                out[entity_codes < 0] = 0.5
            elif key == "cluster_probability":
                out[entity_codes < 0] = 0.0
            elif key == "outlier_score":
                out[entity_codes < 0] = 1.0
            mapped[key] = out

        logger.info(f"  Mapped entity cluster features to {n_tx:,} transactions")
        return mapped
