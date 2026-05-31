"""
Visualization Module
=====================
Plotting utilities for cluster analysis, risk scores, RL training
curves, drift detection timelines, and model comparison.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set global style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.figsize"] = (12, 6)


class FraudVisualizer:
    """
    Visualization utilities for the fraud detection pipeline.

    Parameters
    ----------
    save_dir : str
        Directory to save generated plots.
    """

    def __init__(self, save_dir: str = "experiments/results/plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_cluster_scatter(
        self,
        embedding_2d: np.ndarray,
        cluster_labels: np.ndarray,
        fraud_labels: Optional[np.ndarray] = None,
        title: str = "Behavioral Cluster Visualization",
        filename: str = "cluster_scatter.png",
    ) -> str:
        """
        Plot 2D embedding colored by clusters, with optional fraud overlay.

        Returns path to saved figure.
        """
        fig, axes = plt.subplots(1, 2 if fraud_labels is not None else 1, figsize=(18, 7))

        if fraud_labels is None:
            axes = [axes]

        # Plot 1: Colored by cluster
        ax = axes[0]
        unique_clusters = sorted(set(cluster_labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            label = f"Noise ({mask.sum()})" if cluster == -1 else f"Cluster {cluster} ({mask.sum()})"
            color = "gray" if cluster == -1 else colors[i]
            alpha = 0.3 if cluster == -1 else 0.6
            ax.scatter(
                embedding_2d[mask, 0], embedding_2d[mask, 1],
                c=[color], s=3, alpha=alpha, label=label,
            )

        ax.set_title("Clusters", fontsize=14, fontweight="bold")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        if len(unique_clusters) <= 15:
            ax.legend(fontsize=7, markerscale=4, loc="best")

        # Plot 2: Colored by fraud label
        if fraud_labels is not None:
            ax2 = axes[1]
            legit_mask = fraud_labels == 0
            fraud_mask = fraud_labels == 1

            ax2.scatter(
                embedding_2d[legit_mask, 0], embedding_2d[legit_mask, 1],
                c="steelblue", s=3, alpha=0.3, label=f"Legitimate ({legit_mask.sum()})",
            )
            ax2.scatter(
                embedding_2d[fraud_mask, 0], embedding_2d[fraud_mask, 1],
                c="crimson", s=8, alpha=0.8, label=f"Fraud ({fraud_mask.sum()})",
            )

            ax2.set_title("Fraud Labels", fontsize=14, fontweight="bold")
            ax2.set_xlabel("PC 1")
            ax2.set_ylabel("PC 2")
            ax2.legend(fontsize=10, markerscale=3)

        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        filepath = os.path.join(self.save_dir, filename)
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Plot saved: {filepath}")
        return filepath

    def plot_risk_distribution(
        self,
        risk_scores: np.ndarray,
        fraud_labels: np.ndarray,
        title: str = "Risk Score Distribution",
        filename: str = "risk_distribution.png",
    ) -> str:
        """Plot risk score distribution comparing fraud vs legitimate."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Histogram
        ax1 = axes[0]
        ax1.hist(
            risk_scores[fraud_labels == 0], bins=50, alpha=0.6,
            color="steelblue", label="Legitimate", density=True,
        )
        ax1.hist(
            risk_scores[fraud_labels == 1], bins=50, alpha=0.7,
            color="crimson", label="Fraud", density=True,
        )
        ax1.set_xlabel("Risk Score", fontsize=12)
        ax1.set_ylabel("Density", fontsize=12)
        ax1.set_title("Risk Score Distribution", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)

        # Box plot
        ax2 = axes[1]
        df_plot = pd.DataFrame({
            "Risk Score": risk_scores,
            "Label": np.where(fraud_labels == 1, "Fraud", "Legitimate"),
        })
        sns.boxplot(data=df_plot, x="Label", y="Risk Score", ax=ax2,
                     palette={"Legitimate": "steelblue", "Fraud": "crimson"})
        ax2.set_title("Risk Score by Label", fontsize=14, fontweight="bold")

        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        filepath = os.path.join(self.save_dir, filename)
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Plot saved: {filepath}")
        return filepath

    def plot_training_curves(
        self,
        training_history: Dict[str, List],
        title: str = "DQN Training Progress",
        filename: str = "training_curves.png",
    ) -> str:
        """Plot RL training curves: reward, loss, epsilon, detection rates."""
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        # 1. Episode rewards
        ax1 = fig.add_subplot(gs[0, 0])
        rewards = training_history.get("episode_rewards", [])
        ax1.plot(rewards, alpha=0.3, color="steelblue")
        if len(rewards) > 10:
            window = min(50, len(rewards) // 3)
            smoothed = pd.Series(rewards).rolling(window).mean()
            ax1.plot(smoothed, color="navy", linewidth=2, label=f"MA({window})")
        ax1.set_title("Episode Reward", fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.legend()

        # 2. Average rewards
        ax2 = fig.add_subplot(gs[0, 1])
        avg_rewards = training_history.get("episode_avg_rewards", [])
        ax2.plot(avg_rewards, alpha=0.3, color="seagreen")
        if len(avg_rewards) > 10:
            window = min(50, len(avg_rewards) // 3)
            smoothed = pd.Series(avg_rewards).rolling(window).mean()
            ax2.plot(smoothed, color="darkgreen", linewidth=2, label=f"MA({window})")
        ax2.set_title("Avg Reward per Step", fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.legend()

        # 3. Training loss
        ax3 = fig.add_subplot(gs[0, 2])
        losses = training_history.get("losses", [])
        if losses:
            ax3.plot(losses, alpha=0.4, color="coral")
            if len(losses) > 10:
                window = min(50, len(losses) // 3)
                smoothed = pd.Series(losses).rolling(window).mean()
                ax3.plot(smoothed, color="darkred", linewidth=2, label=f"MA({window})")
            ax3.set_yscale("log")
        ax3.set_title("Training Loss", fontweight="bold")
        ax3.set_xlabel("Episode")
        ax3.legend()

        # 4. Epsilon decay
        ax4 = fig.add_subplot(gs[1, 0])
        epsilons = training_history.get("epsilon_values", [])
        ax4.plot(epsilons, color="purple", linewidth=2)
        ax4.set_title("Epsilon (Exploration)", fontweight="bold")
        ax4.set_xlabel("Episode")
        ax4.set_ylim(-0.05, 1.05)

        # 5. Fraud detection rate
        ax5 = fig.add_subplot(gs[1, 1])
        fdr = training_history.get("fraud_detection_rates", [])
        ax5.plot(fdr, alpha=0.3, color="green")
        if len(fdr) > 10:
            window = min(50, len(fdr) // 3)
            smoothed = pd.Series(fdr).rolling(window).mean()
            ax5.plot(smoothed, color="darkgreen", linewidth=2, label=f"MA({window})")
        ax5.set_title("Fraud Detection Rate", fontweight="bold")
        ax5.set_xlabel("Episode")
        ax5.set_ylim(-0.05, 1.05)
        ax5.legend()

        # 6. False positive rate
        ax6 = fig.add_subplot(gs[1, 2])
        fpr = training_history.get("false_positive_rates", [])
        ax6.plot(fpr, alpha=0.3, color="red")
        if len(fpr) > 10:
            window = min(50, len(fpr) // 3)
            smoothed = pd.Series(fpr).rolling(window).mean()
            ax6.plot(smoothed, color="darkred", linewidth=2, label=f"MA({window})")
        ax6.set_title("False Positive Rate", fontweight="bold")
        ax6.set_xlabel("Episode")
        ax6.set_ylim(-0.05, 1.05)
        ax6.legend()

        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

        filepath = os.path.join(self.save_dir, filename)
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Plot saved: {filepath}")
        return filepath

    def plot_drift_timeline(
        self,
        drift_events: List[Dict],
        total_observations: int,
        title: str = "Concept Drift Detection Timeline",
        filename: str = "drift_timeline.png",
    ) -> str:
        """Plot timeline of detected drift events."""
        fig, ax = plt.subplots(figsize=(14, 5))

        if not drift_events:
            ax.text(0.5, 0.5, "No drift events detected",
                    ha="center", va="center", fontsize=14, transform=ax.transAxes)
        else:
            signal_names = list(set(e["signal"] for e in drift_events))
            colors = plt.cm.Set2(np.linspace(0, 1, len(signal_names)))
            color_map = dict(zip(signal_names, colors))

            for event in drift_events:
                ax.axvline(
                    event["observation"],
                    color=color_map[event["signal"]],
                    alpha=0.7,
                    linewidth=2,
                    label=event["signal"],
                )

            # Deduplicate legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=10)

        ax.set_xlim(0, total_observations)
        ax.set_xlabel("Observation Index", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, filename)
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Plot saved: {filepath}")
        return filepath

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        filename: str = "confusion_matrix.png",
    ) -> str:
        """Plot confusion matrix heatmap."""
        from sklearn.metrics import confusion_matrix as cm_func

        if labels is None:
            labels = ["Legitimate", "Fraud"]

        cm = cm_func(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=ax, cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, filename)
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Plot saved: {filepath}")
        return filepath

    def plot_roc_pr_curves(
        self,
        curves_data: Dict[str, Tuple],
        auc_roc: float = 0.0,
        auc_pr: float = 0.0,
        title: str = "ROC & PR Curves",
        filename: str = "roc_pr_curves.png",
    ) -> str:
        """Plot ROC and Precision-Recall curves side by side."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ROC curve
        ax1 = axes[0]
        fpr, tpr, _ = curves_data["roc"]
        ax1.plot(fpr, tpr, color="darkorange", linewidth=2,
                 label=f"AUC = {auc_roc:.4f}")
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve", fontweight="bold")
        ax1.legend(fontsize=12)
        ax1.set_xlim(-0.02, 1.02)
        ax1.set_ylim(-0.02, 1.02)

        # PR curve
        ax2 = axes[1]
        precision, recall, _ = curves_data["pr"]
        ax2.plot(recall, precision, color="darkgreen", linewidth=2,
                 label=f"AUC = {auc_pr:.4f}")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve", fontweight="bold")
        ax2.legend(fontsize=12)
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(-0.02, 1.02)

        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        filepath = os.path.join(self.save_dir, filename)
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Plot saved: {filepath}")
        return filepath

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20,
        title: str = "Feature Importance",
        filename: str = "feature_importance.png",
    ) -> str:
        """Plot horizontal bar chart of feature importances."""
        # Sort and take top N
        idx = np.argsort(importances)[-top_n:]
        top_features = [feature_names[i] for i in idx]
        top_importances = importances[idx]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        ax.barh(top_features, top_importances, color=colors)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, filename)
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Plot saved: {filepath}")
        return filepath
