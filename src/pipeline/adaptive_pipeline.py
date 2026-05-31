"""
Adaptive Fraud Detection Pipeline
===================================
End-to-end orchestration of all 5 stages: data preprocessing,
behavioral clustering, anomaly detection, RL decision-making,
and concept drift monitoring. Provides train, predict, and
adapt interfaces.
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any

from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.data.feature_engineering import FeatureEngineer
from src.clustering.pca_reducer import PCAReducer
from src.clustering.behavior_aggregator import BehaviorAggregator
from src.clustering.hdbscan_clustering import BehavioralClusterer
from src.models.fraud_estimator import FraudEstimator
from src.anomaly.isolation_forest import AnomalyDetector
from src.anomaly.risk_scorer import RiskScorer
from src.rl.environment import FraudDetectionEnv
from src.rl.dqn_agent import DQNAgent
from src.drift.adwin_detector import ADWINDriftDetector
from src.utils.metrics import FraudMetrics
from src.utils.visualization import FraudVisualizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdaptiveFraudPipeline:
    """
    End-to-end adaptive fraud detection pipeline.

    Chains all five stages together:
    1. Data loading & preprocessing
    2. HDBSCAN behavioral clustering
    3. Isolation Forest anomaly detection & risk scoring
    4. DQN reinforcement learning decision agent
    5. ADWIN concept drift detection

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.results_dir = self.config.get("logging", {}).get("save_dir", "experiments/results/")
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize all pipeline components
        self._init_components()

        # Storage for intermediate results
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_embedding = None
        self.train_risk_scores = None
        self.train_fraud_probs = None
        self.train_cluster_features = None
        self.behavior_train = None
        self.behavior_test = None

        logger.info("=" * 60)
        logger.info("ADAPTIVE FRAUD DETECTION PIPELINE INITIALIZED")
        logger.info("=" * 60)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Config loaded from: {config_path}")
        else:
            logger.warning(f"Config not found at {config_path}, using defaults.")
            config = {}
        return config

    def _init_components(self) -> None:
        """Initialize all pipeline components from config."""
        data_cfg = self.config.get("data", {})
        cluster_cfg = self.config.get("clustering", {})
        supervised_cfg = self.config.get("supervised", {})
        anomaly_cfg = self.config.get("anomaly", {})
        rl_cfg = self.config.get("rl", {})
        drift_cfg = self.config.get("drift", {})

        # Stage 1: Data
        self.loader = DataLoader(
            transaction_path=data_cfg.get("transaction_path", "data/raw/train_transaction.csv"),
            identity_path=data_cfg.get("identity_path", "data/raw/train_identity.csv"),
        )
        self.preprocessor = Preprocessor(
            missing_threshold=data_cfg.get("missing_threshold", 0.7),
            random_state=data_cfg.get("random_state", 42),
        )
        self.feature_engineer = FeatureEngineer(
            user_id_cols=data_cfg.get("user_id_cols", ["card1", "addr1"]),
        )

        # Stage 2: Behavior aggregation -> PCA -> HDBSCAN on behavioral entities
        self.behavior_aggregator = BehaviorAggregator(
            entity_id_cols=cluster_cfg.get(
                "entity_id_cols",
                ["card1", "addr1", "DeviceType", "P_emaildomain"],
            ),
        )
        self.embedding_reducer = PCAReducer(
            n_components=cluster_cfg.get("pca_n_components", 20),
            random_state=cluster_cfg.get("random_state", 42),
        )
        self.clusterer = BehavioralClusterer(
            min_cluster_size=cluster_cfg.get("hdbscan_min_cluster_size", 30),
            min_samples=cluster_cfg.get("hdbscan_min_samples", 15),
            metric=cluster_cfg.get("hdbscan_metric", "euclidean"),
            cluster_selection_method=cluster_cfg.get("hdbscan_cluster_selection_method", "eom"),
        )

        # Stage 2b: Supervised fraud estimator
        self.supervised_enabled = supervised_cfg.get("enabled", True)
        if self.supervised_enabled:
            self.fraud_estimator = FraudEstimator(
                n_estimators=supervised_cfg.get("n_estimators", 500),
                learning_rate=supervised_cfg.get("learning_rate", 0.05),
                num_leaves=supervised_cfg.get("num_leaves", 64),
                random_state=data_cfg.get("random_state", 42),
                early_stopping_rounds=supervised_cfg.get("early_stopping_rounds", 50),
            )
        else:
            self.fraud_estimator = None

        # Stage 3: Anomaly Detection
        self.anomaly_detector = AnomalyDetector(
            contamination=anomaly_cfg.get("isolation_forest_contamination", 0.035),
            n_estimators=anomaly_cfg.get("n_estimators", 200),
            random_state=anomaly_cfg.get("random_state", 42),
        )
        risk_weights = anomaly_cfg.get("risk_weights", None)
        self.risk_scorer = RiskScorer(initial_weights=risk_weights)

        # Stage 4: RL Agent (initialized later with state_dim)
        self.rl_config = rl_cfg
        self.rl_agent = None  # Created after we know state_dim

        # Stage 5: Drift Detection
        self.drift_detector = ADWINDriftDetector(
            delta=drift_cfg.get("adwin_delta", 0.002),
            monitored_signals=drift_cfg.get("monitored_signals", None),
            drift_check_interval=drift_cfg.get("drift_check_interval", 1000),
        )

        # Visualization
        self.visualizer = FraudVisualizer(
            save_dir=os.path.join(self.results_dir, "plots"),
        )

    def _fit_clustering(
        self,
        X: np.ndarray,
        behavior_ids: pd.Series,
        y: Optional[pd.Series] = None,
    ) -> Dict[str, np.ndarray]:
        """Behavior aggregation -> PCA -> HDBSCAN on entities, mapped to transactions."""
        y_arr = y.values if y is not None else None

        entity_features, entity_codes, entity_fraud = (
            self.behavior_aggregator.aggregate_entities(X, behavior_ids, y=y_arr)
        )
        embedding = self.embedding_reducer.fit_transform(entity_features)
        self.train_embedding = embedding

        entity_cluster = self.clusterer.fit_predict(embedding, labels=entity_fraud)
        return self.behavior_aggregator.map_entity_results_to_transactions(
            entity_cluster, entity_codes
        )

    def _predict_clustering(
        self,
        X: np.ndarray,
        behavior_ids: pd.Series,
    ) -> Dict[str, np.ndarray]:
        """Cluster held-out or inference data using fitted reducers."""
        entity_features, entity_codes, _ = self.behavior_aggregator.aggregate_entities(
            X, behavior_ids, y=None
        )
        embedding = self.embedding_reducer.transform(entity_features)
        entity_cluster = self.clusterer.predict(embedding)
        return self.behavior_aggregator.map_entity_results_to_transactions(
            entity_cluster, entity_codes
        )

    def _build_supervised_matrix(
        self,
        X: np.ndarray,
        cluster_results: Dict[str, np.ndarray],
        risk_scores: np.ndarray,
    ) -> np.ndarray:
        """Feature matrix for LightGBM: base features + cluster + risk signals."""
        cluster_block = FraudEstimator.cluster_feature_matrix(cluster_results)
        return np.hstack([X, cluster_block, risk_scores.reshape(-1, 1)]).astype(np.float32)

    # ==================================================================
    # MAIN PIPELINE METHODS
    # ==================================================================

    def train(self, sample_frac: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute the full training pipeline.

        Parameters
        ----------
        sample_frac : float, optional
            Fraction of data to use (for faster development).

        Returns
        -------
        dict
            Comprehensive training results and metrics.
        """
        logger.info("\n" + "=" * 70)
        logger.info("   STARTING FULL TRAINING PIPELINE")
        logger.info("=" * 70 + "\n")

        results = {}

        # ---- Stage 1: Load & Preprocess ----
        logger.info(">> STAGE 1: Data Loading & Preprocessing")
        raw_data = self.loader.load(sample_frac=sample_frac)
        processed_data = self.preprocessor.fit_transform(raw_data)
        engineered_data = self.feature_engineer.fit_transform(processed_data)
        behavior_ids = self.behavior_aggregator.build_behavior_id(engineered_data)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = (
            self.preprocessor.split_data(engineered_data)
        )
        self.behavior_train = behavior_ids.loc[self.X_train.index]
        self.behavior_test = behavior_ids.loc[self.X_test.index]

        results["data"] = {
            "total_samples": len(engineered_data),
            "train_samples": len(self.X_train),
            "test_samples": len(self.X_test),
            "n_features": self.X_train.shape[1],
            "fraud_rate": float(self.y_train.mean()),
        }
        logger.info(f"  Data ready: {results['data']}")

        # ---- Stage 2: Behavior Aggregation -> PCA -> HDBSCAN ----
        logger.info("\n>> STAGE 2: Behavioral Clustering (aggregate -> PCA -> HDBSCAN)")
        X_train_values = self.X_train.values.astype(np.float32)
        X_train_clean = np.nan_to_num(X_train_values, nan=0.0, posinf=0.0, neginf=0.0)

        cluster_results = self._fit_clustering(
            X_train_clean, self.behavior_train, y=self.y_train
        )
        self.train_cluster_features = FraudEstimator.cluster_feature_matrix(
            cluster_results
        )

        results["clustering"] = {
            "n_clusters": len(set(cluster_results["cluster_label"])) - (
                1 if -1 in cluster_results["cluster_label"] else 0
            ),
            "noise_fraction": float((cluster_results["cluster_label"] == -1).mean()),
        }

        # 2D PCA scatter for cluster / fraud visualization
        try:
            embedding_2d = PCAReducer.fit_transform_2d(X_train_clean)
            self.visualizer.plot_cluster_scatter(
                embedding_2d,
                cluster_results["cluster_label"],
                fraud_labels=self.y_train.values,
            )
        except Exception as e:
            logger.warning(f"  Could not generate cluster visualization: {e}")

        # ---- Stage 3: Anomaly Detection & Risk Scoring ----
        logger.info("\n>> STAGE 3: Anomaly Detection & Risk Scoring")
        anomaly_results = self.anomaly_detector.fit_predict(X_train_clean)

        # Compute composite risk scores
        amount_zscore = None
        if "user_amount_zscore" in self.X_train.columns:
            amount_zscore = self.X_train["user_amount_zscore"].values
        elif "global_amount_zscore" in self.X_train.columns:
            amount_zscore = self.X_train["global_amount_zscore"].values

        velocity_score = None
        if "tx_velocity_24h" in self.X_train.columns:
            velocity_score = self.X_train["tx_velocity_24h"].values

        risk_results = self.risk_scorer.compute_risk_score(
            anomaly_score=anomaly_results["anomaly_score_normalized"],
            outlier_score=cluster_results["outlier_score"],
            cluster_probability=cluster_results["cluster_probability"],
            centroid_distance=cluster_results["centroid_distance"],
            cluster_fraud_rate=cluster_results["cluster_fraud_rate"],
            amount_zscore=amount_zscore,
            velocity_score=velocity_score,
        )

        self.train_risk_scores = risk_results["risk_score"]

        # Calibrate risk scorer with supervised labels
        self.risk_scorer.calibrate(risk_results["risk_signals"], self.y_train)

        # Recompute calibrated risk scores
        risk_results_calibrated = self.risk_scorer.compute_risk_score(
            anomaly_score=anomaly_results["anomaly_score_normalized"],
            outlier_score=cluster_results["outlier_score"],
            cluster_probability=cluster_results["cluster_probability"],
            centroid_distance=cluster_results["centroid_distance"],
            cluster_fraud_rate=cluster_results["cluster_fraud_rate"],
            amount_zscore=amount_zscore,
            velocity_score=velocity_score,
        )
        self.train_risk_scores = risk_results_calibrated["risk_score"]

        # Visualize risk distribution
        try:
            self.visualizer.plot_risk_distribution(
                self.train_risk_scores, self.y_train.values
            )
        except Exception as e:
            logger.warning(f"  Could not generate risk distribution plot: {e}")

        # Binary evaluation of risk scores
        risk_pred = (self.train_risk_scores > 0.5).astype(int)
        risk_metrics = FraudMetrics.binary_metrics(
            self.y_train.values, risk_pred, self.train_risk_scores
        )
        results["risk_scoring"] = risk_metrics
        FraudMetrics.print_report(risk_metrics, "Risk Score Evaluation (Train)")

        # ---- Stage 3b: Supervised LightGBM fraud estimator ----
        if self.fraud_estimator is not None:
            logger.info("\n>> STAGE 3b: Supervised Fraud Estimator (LightGBM)")
            X_sup_train = self._build_supervised_matrix(
                X_train_clean, cluster_results, self.train_risk_scores
            )
            X_test_clean_pre = np.nan_to_num(
                self.X_test.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            test_cluster_pre = self._predict_clustering(
                X_test_clean_pre, self.behavior_test
            )
            test_anom_pre = self.anomaly_detector.predict(X_test_clean_pre)
            test_risk_pre = self.risk_scorer.compute_risk_score(
                anomaly_score=test_anom_pre["anomaly_score_normalized"],
                outlier_score=test_cluster_pre.get(
                    "outlier_score", np.zeros(len(X_test_clean_pre))
                ),
                cluster_probability=test_cluster_pre["cluster_probability"],
                centroid_distance=test_cluster_pre["centroid_distance"],
                cluster_fraud_rate=test_cluster_pre["cluster_fraud_rate"],
            )["risk_score"]
            X_sup_test = self._build_supervised_matrix(
                X_test_clean_pre, test_cluster_pre, test_risk_pre
            )

            lgbm_metrics = self.fraud_estimator.fit(
                X_sup_train,
                self.y_train.values,
                X_val=X_sup_test,
                y_val=self.y_test.values,
            )
            self.train_fraud_probs = self.fraud_estimator.predict_proba(X_sup_train)
            results["supervised"] = lgbm_metrics

            lgbm_train_metrics = FraudMetrics.binary_metrics(
                self.y_train.values,
                (self.train_fraud_probs > 0.5).astype(int),
                self.train_fraud_probs,
            )
            FraudMetrics.print_report(lgbm_train_metrics, "LightGBM (Train)")
        else:
            self.train_fraud_probs = np.zeros(len(X_train_clean), dtype=np.float32)

        # ---- Stage 4: RL Training ----
        logger.info("\n>> STAGE 4: Reinforcement Learning Training")
        n_cluster_feats = self.train_cluster_features.shape[1]
        state_dim = self.X_train.shape[1] + 1 + 1 + n_cluster_feats

        self.rl_agent = DQNAgent(
            state_dim=state_dim,
            n_actions=self.rl_config.get("num_actions", 3),
            learning_rate=self.rl_config.get("learning_rate", 0.001),
            gamma=self.rl_config.get("gamma", 0.99),
            epsilon_start=self.rl_config.get("epsilon_start", 1.0),
            epsilon_end=self.rl_config.get("epsilon_end", 0.01),
            epsilon_decay=self.rl_config.get("epsilon_decay", 0.995),
            batch_size=self.rl_config.get("batch_size", 64),
            buffer_size=self.rl_config.get("buffer_size", 50000),
            target_update_freq=self.rl_config.get("target_update_freq", 100),
            tau=self.rl_config.get("tau", 0.005),
            hidden_dims=self.rl_config.get("hidden_dims", [128, 128, 64]),
            dropout=self.rl_config.get("dropout", 0.2),
        )

        # Create training environment
        rewards_config = self.rl_config.get("rewards", None)
        train_env = FraudDetectionEnv(
            X=X_train_clean,
            y=self.y_train.values,
            risk_scores=self.train_risk_scores,
            fraud_probs=self.train_fraud_probs,
            cluster_features=self.train_cluster_features,
            rewards_config=rewards_config,
            max_steps=self.rl_config.get("max_steps_per_episode", 1000),
            shuffle=True,
        )

        # Train the agent
        num_episodes = self.rl_config.get("num_episodes", 500)
        training_history = self.rl_agent.train(
            env=train_env,
            num_episodes=num_episodes,
            log_interval=max(1, num_episodes // 20),
        )

        results["rl_training"] = {
            "final_reward": training_history["episode_rewards"][-1] if training_history["episode_rewards"] else 0,
            "best_reward": max(training_history["episode_rewards"]) if training_history["episode_rewards"] else 0,
            "final_fdr": training_history["fraud_detection_rates"][-1] if training_history["fraud_detection_rates"] else 0,
            "final_fpr": training_history["false_positive_rates"][-1] if training_history["false_positive_rates"] else 0,
            "final_epsilon": self.rl_agent.epsilon,
        }

        # Visualize training curves
        try:
            self.visualizer.plot_training_curves(training_history)
        except Exception as e:
            logger.warning(f"  Could not generate training curves: {e}")

        # ---- Evaluate on test set ----
        logger.info("\n>> TEST SET EVALUATION")
        test_results = self._evaluate_test_set()
        results["test_evaluation"] = test_results

        # ---- Save models ----
        self._save_all_models()

        logger.info("\n" + "=" * 70)
        logger.info("   TRAINING PIPELINE COMPLETE")
        logger.info("=" * 70)

        return results

    def _evaluate_test_set(self) -> Dict:
        """Evaluate the full pipeline on the test set."""
        if self.X_test is None or self.rl_agent is None:
            logger.warning("Pipeline not trained yet. Skipping test evaluation.")
            return {}

        X_test_values = self.X_test.values.astype(np.float32)
        X_test_clean = np.nan_to_num(X_test_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Stage 2: Cluster test data
        cluster_results = self._predict_clustering(X_test_clean, self.behavior_test)
        test_cluster_features = FraudEstimator.cluster_feature_matrix(cluster_results)

        # Stage 3: Anomaly detection on test
        anomaly_results = self.anomaly_detector.predict(X_test_clean)

        # Compute risk scores for test
        amount_zscore = None
        if "user_amount_zscore" in self.X_test.columns:
            amount_zscore = self.X_test["user_amount_zscore"].values
        elif "global_amount_zscore" in self.X_test.columns:
            amount_zscore = self.X_test["global_amount_zscore"].values

        velocity_score = None
        if "tx_velocity_24h" in self.X_test.columns:
            velocity_score = self.X_test["tx_velocity_24h"].values

        risk_results = self.risk_scorer.compute_risk_score(
            anomaly_score=anomaly_results["anomaly_score_normalized"],
            outlier_score=cluster_results.get("outlier_score", np.zeros(len(X_test_clean))),
            cluster_probability=cluster_results["cluster_probability"],
            centroid_distance=cluster_results["centroid_distance"],
            cluster_fraud_rate=cluster_results["cluster_fraud_rate"],
            amount_zscore=amount_zscore,
            velocity_score=velocity_score,
        )

        test_risk_scores = risk_results["risk_score"]

        if self.fraud_estimator is not None:
            X_sup_test = self._build_supervised_matrix(
                X_test_clean, cluster_results, test_risk_scores
            )
            test_fraud_probs = self.fraud_estimator.predict_proba(X_sup_test)
            lgbm_metrics = FraudMetrics.binary_metrics(
                self.y_test.values,
                (test_fraud_probs > 0.5).astype(int),
                test_fraud_probs,
            )
            FraudMetrics.print_report(lgbm_metrics, "LightGBM (Test)")
            score_for_curves = test_fraud_probs
        else:
            test_fraud_probs = np.zeros(len(X_test_clean), dtype=np.float32)
            score_for_curves = test_risk_scores

        # Stage 4: RL evaluation
        test_env = FraudDetectionEnv(
            X=X_test_clean,
            y=self.y_test.values,
            risk_scores=test_risk_scores,
            fraud_probs=test_fraud_probs,
            cluster_features=test_cluster_features,
            max_steps=len(X_test_clean),
            shuffle=False,
        )

        eval_results = self.rl_agent.evaluate(test_env, n_episodes=3)

        risk_pred = (test_risk_scores > 0.5).astype(int)
        risk_metrics = FraudMetrics.binary_metrics(
            self.y_test.values, risk_pred, test_risk_scores
        )
        FraudMetrics.print_report(risk_metrics, "Risk Score Evaluation (Test)")

        try:
            curves = FraudMetrics.get_curves(self.y_test.values, score_for_curves)
            primary_metrics = (
                lgbm_metrics if self.fraud_estimator is not None else risk_metrics
            )
            self.visualizer.plot_roc_pr_curves(
                curves,
                auc_roc=primary_metrics.get("auc_roc", 0),
                auc_pr=primary_metrics.get("auc_pr", 0),
            )
        except Exception as e:
            logger.warning(f"  Could not generate ROC/PR curves: {e}")

        return {
            "rl_evaluation": eval_results,
            "risk_binary_metrics": risk_metrics,
            "supervised_binary_metrics": (
                lgbm_metrics if self.fraud_estimator is not None else None
            ),
        }

    def predict(self, transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict fraud decisions for new transactions.

        Parameters
        ----------
        transaction_data : pd.DataFrame
            Raw transaction data (same format as training data).

        Returns
        -------
        dict with 'decisions', 'risk_scores', 'risk_categories'
        """
        if self.rl_agent is None:
            raise RuntimeError("Pipeline not trained. Call train() first.")

        # Preprocess
        processed = self.preprocessor.transform(transaction_data)
        engineered = self.feature_engineer.transform(processed)

        exclude_cols = {"isFraud", "TransactionID", "TransactionDT"}
        feature_cols = [c for c in engineered.columns if c not in exclude_cols]
        X = engineered[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        behavior_ids = self.behavior_aggregator.build_behavior_id(engineered)
        cluster_results = self._predict_clustering(X, behavior_ids)
        cluster_block = FraudEstimator.cluster_feature_matrix(cluster_results)

        # Anomaly detection
        anomaly_results = self.anomaly_detector.predict(X)

        # Risk scoring
        risk_results = self.risk_scorer.compute_risk_score(
            anomaly_score=anomaly_results["anomaly_score_normalized"],
            outlier_score=cluster_results.get("outlier_score", np.zeros(len(X))),
            cluster_probability=cluster_results["cluster_probability"],
            centroid_distance=cluster_results["centroid_distance"],
            cluster_fraud_rate=cluster_results["cluster_fraud_rate"],
        )

        risk_scores = risk_results["risk_score"]

        if self.fraud_estimator is not None:
            X_sup = self._build_supervised_matrix(X, cluster_results, risk_scores)
            fraud_probs = self.fraud_estimator.predict_proba(X_sup)
        else:
            fraud_probs = np.zeros(len(X), dtype=np.float32)

        # RL decisions
        decisions = []
        for i in range(len(X)):
            state = np.concatenate([
                X[i], [risk_scores[i]], [fraud_probs[i]], cluster_block[i],
            ]).astype(np.float32)
            state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
            action = self.rl_agent.select_action(state, training=False)
            decisions.append(action)

        decisions = np.array(decisions)
        decision_names = np.array([
            FraudDetectionEnv.ACTION_NAMES[a] for a in decisions
        ])

        return {
            "decisions": decisions,
            "decision_names": decision_names,
            "risk_scores": risk_scores,
            "fraud_probabilities": fraud_probs,
            "risk_categories": risk_results["risk_category"],
            "cluster_labels": cluster_results["cluster_label"],
            "anomaly_scores": anomaly_results["anomaly_score_normalized"],
        }

    def update_with_feedback(
        self,
        transaction_features: np.ndarray,
        risk_score: float,
        action_taken: int,
        actual_fraud: int,
    ) -> bool:
        """
        Process feedback from a resolved transaction to update the RL agent.

        Parameters
        ----------
        transaction_features : np.ndarray
            Feature vector of the transaction.
        risk_score : float
            Risk score of the transaction.
        action_taken : int
            Action taken by the agent.
        actual_fraud : int
            True fraud label (0 or 1).

        Returns
        -------
        bool
            True if drift was detected during this update.
        """
        if self.rl_agent is None:
            return False

        cluster_block = np.zeros(5, dtype=np.float32)
        fraud_prob = 0.0
        state = np.concatenate([
            transaction_features, [risk_score], [fraud_prob], cluster_block,
        ]).astype(np.float32)
        reward_map = self.rl_config.get("rewards", {})

        if actual_fraud:
            if action_taken == 2:
                reward = reward_map.get("correct_fraud_block", 10)
            elif action_taken == 1:
                reward = reward_map.get("correct_review_fraud", 5)
            else:
                reward = reward_map.get("missed_fraud", -25)
        else:
            if action_taken == 0:
                reward = reward_map.get("correct_approve", 3)
            elif action_taken == 1:
                reward = reward_map.get("review_cost", -1)
            else:
                reward = reward_map.get("false_positive_block", -15)

        # Store experience (next_state = current state for simplicity)
        self.rl_agent.replay_buffer.push(state, action_taken, reward, state, False)

        # Train step
        self.rl_agent.train_step()

        # Check for concept drift
        prediction_error = 1.0 if (
            (actual_fraud and action_taken == 0) or
            (not actual_fraud and action_taken == 2)
        ) else 0.0

        drift_detected = self.drift_detector.check_and_adapt(
            signal_values={
                "prediction_error": prediction_error,
                "risk_score": risk_score,
            },
            adaptation_callbacks={
                "prediction_error": lambda: self._handle_drift(),
            },
        )

        return drift_detected

    def _handle_drift(self) -> None:
        """Handle detected concept drift by resetting exploration."""
        drift_cfg = self.config.get("drift", {})
        reset_epsilon = drift_cfg.get("epsilon_reset_value", 0.3)

        logger.warning("  [!] Concept drift detected! Initiating adaptation...")
        logger.info(f"  Resetting RL epsilon to {reset_epsilon} for re-exploration")

        if self.rl_agent:
            self.rl_agent.set_epsilon(reset_epsilon)

    def _save_all_models(self) -> None:
        """Save all fitted models to disk."""
        models_dir = os.path.join(self.results_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        try:
            self.embedding_reducer.save(os.path.join(models_dir, "pca_reducer.pkl"))
        except Exception as e:
            logger.warning(f"  Could not save embedding reducer: {e}")

        try:
            self.clusterer.save(os.path.join(models_dir, "hdbscan_clusterer.pkl"))
        except Exception as e:
            logger.warning(f"  Could not save clusterer: {e}")

        try:
            self.anomaly_detector.save(os.path.join(models_dir, "isolation_forest.pkl"))
        except Exception as e:
            logger.warning(f"  Could not save anomaly detector: {e}")

        try:
            self.risk_scorer.save(os.path.join(models_dir, "risk_scorer.pkl"))
        except Exception as e:
            logger.warning(f"  Could not save risk scorer: {e}")

        try:
            if self.fraud_estimator and self.fraud_estimator.is_fitted:
                self.fraud_estimator.save(
                    os.path.join(models_dir, "fraud_estimator.pkl")
                )
        except Exception as e:
            logger.warning(f"  Could not save fraud estimator: {e}")

        try:
            if self.rl_agent:
                self.rl_agent.save(os.path.join(models_dir, "dqn_agent.pth"))
        except Exception as e:
            logger.warning(f"  Could not save RL agent: {e}")

        logger.info(f"  All models saved to: {models_dir}")

    def get_drift_summary(self) -> Dict:
        """Get a summary of all detected drift events."""
        return self.drift_detector.get_drift_summary()


# ==================================================================
# CLI Entry Point
# ==================================================================

def main():
    """Run the full pipeline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Adaptive E-Commerce Fraud Detection Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--sample-size", type=float, default=None,
        help="Fraction of data to use (0-1) for faster development",
    )
    args = parser.parse_args()

    pipeline = AdaptiveFraudPipeline(config_path=args.config)
    results = pipeline.train(sample_frac=args.sample_size)

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)

    for stage_name, stage_results in results.items():
        logger.info(f"\n  {stage_name}:")
        if isinstance(stage_results, dict):
            for key, value in stage_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {key}: {value}")
                elif isinstance(value, dict):
                    for k2, v2 in value.items():
                        if isinstance(v2, (int, float)):
                            logger.info(f"    {key}.{k2}: {v2}")


if __name__ == "__main__":
    main()
