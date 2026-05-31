"""
Evaluation Metrics Module
==========================
Comprehensive fraud detection metrics including standard classification
metrics, cost-sensitive evaluation, and 3-class confusion analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FraudMetrics:
    """
    Compute and report fraud detection evaluation metrics.

    Handles both binary (fraud/legit) and 3-class (approve/review/block)
    evaluation scenarios.
    """

    @staticmethod
    def binary_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute binary classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True labels (0/1).
        y_pred : np.ndarray
            Predicted labels (0/1).
        y_prob : np.ndarray, optional
            Predicted probabilities for the positive class.

        Returns
        -------
        dict
            Dictionary of metric names and values.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_prob is not None:
            try:
                metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics["auc_roc"] = 0.0
            try:
                metrics["auc_pr"] = average_precision_score(y_true, y_prob)
            except ValueError:
                metrics["auc_pr"] = 0.0

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["false_positive_rate"] = fp / max(1, fp + tn)
        metrics["false_negative_rate"] = fn / max(1, fn + tp)

        return metrics

    @staticmethod
    def three_class_metrics(
        y_true_fraud: np.ndarray,
        actions: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate 3-action fraud decisions (approve/review/block).

        Parameters
        ----------
        y_true_fraud : np.ndarray
            True fraud labels (0/1).
        actions : np.ndarray
            Agent actions (0=approve, 1=review, 2=block).

        Returns
        -------
        dict
            Comprehensive 3-class metrics.
        """
        n = len(y_true_fraud)
        fraud_mask = y_true_fraud == 1
        legit_mask = y_true_fraud == 0

        # Action breakdowns
        n_approve = (actions == 0).sum()
        n_review = (actions == 1).sum()
        n_block = (actions == 2).sum()

        # Fraud handling
        fraud_approved = ((actions == 0) & fraud_mask).sum()
        fraud_reviewed = ((actions == 1) & fraud_mask).sum()
        fraud_blocked = ((actions == 2) & fraud_mask).sum()

        # Legitimate handling
        legit_approved = ((actions == 0) & legit_mask).sum()
        legit_reviewed = ((actions == 1) & legit_mask).sum()
        legit_blocked = ((actions == 2) & legit_mask).sum()

        n_fraud = fraud_mask.sum()
        n_legit = legit_mask.sum()

        metrics = {
            # Action distribution
            "approve_rate": n_approve / n,
            "review_rate": n_review / n,
            "block_rate": n_block / n,
            # Fraud detection
            "fraud_caught_rate": (fraud_blocked + fraud_reviewed) / max(1, n_fraud),
            "fraud_block_rate": fraud_blocked / max(1, n_fraud),
            "fraud_review_rate": fraud_reviewed / max(1, n_fraud),
            "fraud_missed_rate": fraud_approved / max(1, n_fraud),
            # Legitimate handling
            "legit_approve_rate": legit_approved / max(1, n_legit),
            "legit_review_rate": legit_reviewed / max(1, n_legit),
            "legit_block_rate": legit_blocked / max(1, n_legit),
            # Overall
            "correct_decisions": (legit_approved + fraud_blocked) / n,
            "total_transactions": n,
            "total_fraud": int(n_fraud),
            "total_legitimate": int(n_legit),
        }

        return metrics

    @staticmethod
    def cost_sensitive_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        avg_fraud_amount: float = 100.0,
        review_cost: float = 5.0,
        false_positive_cost: float = 10.0,
    ) -> Dict[str, float]:
        """
        Cost-sensitive evaluation of fraud detection.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels or actions.
        avg_fraud_amount : float
            Average financial loss per missed fraud.
        review_cost : float
            Cost of manual review per transaction.
        false_positive_cost : float
            Cost per false positive (customer friction).

        Returns
        -------
        dict
            Financial cost metrics.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, (y_pred > 0).astype(int), labels=[0, 1]).ravel()

        fraud_loss = fn * avg_fraud_amount
        review_expense = (y_pred == 1).sum() * review_cost if hasattr(y_pred, '__len__') else 0
        fp_cost = fp * false_positive_cost

        total_cost = fraud_loss + review_expense + fp_cost
        prevented_loss = tp * avg_fraud_amount

        return {
            "total_cost": total_cost,
            "fraud_loss": fraud_loss,
            "review_expense": review_expense,
            "false_positive_cost": fp_cost,
            "prevented_loss": prevented_loss,
            "net_savings": prevented_loss - total_cost,
            "cost_per_transaction": total_cost / max(1, len(y_true)),
        }

    @staticmethod
    def get_curves(
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, Tuple]:
        """
        Compute ROC and PR curves for plotting.

        Returns
        -------
        dict with 'roc' and 'pr' curve data.
        """
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

        return {
            "roc": (fpr, tpr, roc_thresholds),
            "pr": (precision, recall, pr_thresholds),
        }

    @staticmethod
    def print_report(
        metrics: Dict[str, float],
        title: str = "Evaluation Report",
    ) -> str:
        """Format and print metrics report."""
        lines = [
            f"\n{'=' * 50}",
            f"  {title}",
            f"{'=' * 50}",
        ]

        for key, value in metrics.items():
            if isinstance(value, float):
                if "rate" in key or "accuracy" in key or key.startswith("auc"):
                    lines.append(f"  {key:>30s}: {value:.4f} ({value:.1%})")
                else:
                    lines.append(f"  {key:>30s}: {value:.4f}")
            else:
                lines.append(f"  {key:>30s}: {value}")

        lines.append(f"{'=' * 50}\n")
        report = "\n".join(lines)
        logger.info(report)
        return report
