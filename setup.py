from setuptools import setup, find_packages

setup(
    name="adaptive-fraud-detection",
    version="0.1.0",
    description="Adaptive E-Commerce Fraud Detection using Behavioral Clustering, RL, and Concept Drift Detection",
    author="Research Internship",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.11",
        "hdbscan>=0.8.33",
        "torch>=2.0",
        "lightgbm>=4.0",
        "gymnasium>=0.29",
        "river>=0.21",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "plotly>=5.15",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "joblib>=1.3",
        "category_encoders>=2.6",
    ],
)
