# Adaptive E-Commerce Fraud Detection System

## Behavioral Clustering • Anomaly Detection • Reinforcement Learning • Concept Drift Detection

An intelligent fraud detection framework that continuously adapts to evolving fraud patterns using behavioral analytics, machine learning, reinforcement learning, and concept drift detection.

---

## Overview

Traditional fraud detection systems often rely on static rules or supervised machine learning models. While effective initially, these approaches struggle when fraudsters change their behavior over time.

This project presents an **Adaptive E-Commerce Fraud Detection System** that combines:

- Behavioral Clustering using HDBSCAN
- Anomaly Detection using Isolation Forest
- Fraud Probability Estimation using LightGBM
- Reinforcement Learning using Double Dueling DQN
- Concept Drift Detection using ADWIN

The framework continuously learns from feedback and adapts to emerging fraud patterns, making it suitable for real-world dynamic e-commerce environments.

---

## Key Features

### Behavioral Clustering

- Aggregate transactions into behavioral entities (users, devices, accounts)
- PCA-based dimensionality reduction
- HDBSCAN clustering for behavioral pattern discovery
- Identification of suspicious clusters and outliers

### Fraud Probability Estimation

- LightGBM-based supervised fraud classifier
- Generates calibrated fraud probabilities
- Handles highly imbalanced transaction datasets efficiently

### Composite Risk Scoring

Risk scores combine multiple fraud indicators:

- Isolation Forest anomaly score
- Cluster outlier probability
- Distance from cluster centroid
- Behavioral deviation metrics

### Reinforcement Learning Decision Engine

A Double Dueling Deep Q-Network (DQN) learns optimal actions:

| Action | Description |
|----------|------------|
| APPROVE | Allow transaction |
| REVIEW | Send for manual review |
| BLOCK | Reject transaction |

### Concept Drift Detection

- ADWIN-based drift monitoring
- Continuous detection of evolving fraud patterns
- Automatic model adaptation and retraining

### Synthetic Data Fallback

- Automatic synthetic fraud data generation
- Enables experimentation without requiring external datasets

---

## System Architecture

```text
Raw Transaction Data
        ↓
Stage 1: Preprocessing & Feature Engineering
        ↓
Stage 2: Behavior Aggregation → PCA → HDBSCAN
        ↓
Stage 3: Isolation Forest + Risk Scoring + LightGBM
        ↓
Stage 4: DQN Agent → Approve / Review / Block
        ↓
Stage 5: ADWIN Concept Drift Monitoring
        ↓
Model Adaptation & Feedback Loop
```

## Project Structure

```text
adaptive-ecommerce-fraud-detection/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── src/
│   ├── data/
│   ├── clustering/
│   ├── anomaly/
│   ├── models/
│   ├── rl/
│   ├── drift/
│   ├── pipeline/
│   └── utils/
│
├── experiments/
│   ├── models/
│   ├── results/
│   └── plots/
│
├── tests/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Clustering | HDBSCAN |
| Dimensionality Reduction | PCA |
| Anomaly Detection | Isolation Forest |
| Supervised Learning | LightGBM |
| Reinforcement Learning | PyTorch |
| Drift Detection | River (ADWIN) |
| Testing | PyTest |

---

## Dataset

### IEEE-CIS Fraud Detection Dataset

This project supports the IEEE-CIS Fraud Detection dataset.

Download:

https://www.kaggle.com/c/ieee-fraud-detection/data

Place the dataset files inside:

```text
src/dataset/
```

### Synthetic Dataset Support

If the IEEE-CIS dataset is unavailable, the system automatically generates realistic synthetic transaction data for experimentation and testing.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/saisrinath21/adaptive_fraud_detection.git

cd adaptive_fraud_detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

Run the complete pipeline:

```bash
python -m src.pipeline.adaptive_pipeline --config config/config.yaml
```

Run using a sample of the dataset:

```bash
python -m src.pipeline.adaptive_pipeline \
--config config/config.yaml \
--sample-size 0.1
```

---

## Example Usage

### Train the Pipeline

```python
from src.pipeline.adaptive_pipeline import AdaptiveFraudPipeline

pipeline = AdaptiveFraudPipeline("config/config.yaml")

results = pipeline.train(sample_frac=0.2)
```

### Make Predictions

```python
predictions = pipeline.predict(transaction_df)

print(predictions["decision_names"])
```

Example output:

```python
['APPROVE', 'BLOCK', 'REVIEW']
```

### Online Feedback Learning

```python
pipeline.update_with_feedback(
    transaction_features=features,
    risk_score=0.82,
    action_taken=2,
    actual_fraud=1
)
```

The feedback is used to continuously improve the reinforcement learning policy.

---

## Evaluation Metrics

### Fraud Detection Metrics

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

### Reinforcement Learning Metrics

- Average Reward
- Fraud Loss Reduction
- Decision Accuracy

### Drift Detection Metrics

- Drift Detection Frequency
- Adaptation Speed
- Post-Drift Performance

---

## Research Contributions

This project proposes a hybrid fraud detection architecture that integrates:

1. Behavioral Clustering
2. Anomaly Detection
3. Supervised Fraud Classification
4. Reinforcement Learning-Based Decision Making
5. Online Concept Drift Detection

The system is designed to operate effectively in dynamic fraud environments where attacker behavior continuously evolves.

---

## Future Enhancements

- Graph Neural Networks (GNNs) for fraud ring detection
- Online incremental learning
- Explainable AI using SHAP
- Real-time streaming with Kafka
- Federated fraud detection
- Multi-agent reinforcement learning

---

## Results

Example outputs generated by the framework:

- Behavioral cluster visualizations
- Fraud probability distributions
- Isolation Forest anomaly scores
- RL reward convergence plots
- Concept drift detection timelines
- Precision, Recall, F1, ROC-AUC metrics

Store generated outputs in:

```text
experiments/results/
```

---

## License

This project is intended for academic and research purposes.

See the LICENSE file for more information.

---
