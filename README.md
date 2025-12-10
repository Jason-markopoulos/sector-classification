# Sector Classification

This project develops a complete machine learning pipeline to predict the economic sector of companies based on financial ratios and accounting features.
The workflow includes data preprocessing, exploratory analysis, model training, hyperparameter tuning, evaluation, and interpretability (SHAP, permutation importance, model-specific importance metrics).

The objective is to evaluate whether fundamental financial variables can reliably distinguish between economic sectors - and to understand why models make the predictions they do, using modern interpretability tools.

This repository is structured to be fully reproducible, transparent, and pipeline-driven, following best practices for machine learning research.

# Environment & Reproducibility

All code in this repository was developed and tested using:

Python 3.10

scikit-learn (classification models, preprocessing, CV)

XGBoost (gradient boosting classifier)

SHAP (global/local interpretability)

UMAP (dimensionality reduction)

statsmodels (statistical analysis)

To ensure full reproducibility, the repository includes an environment.yml file specifying all dependencies and sources.


# Summary of Models & Evaluation

The project evaluates several widely used machine learning models:

L1-penalized Logistic Regression

L2-penalized Logistic Regression

Elastic Net Logistic Regression

Random Forest

Support Vector Machine (RBF kernel)

XGBoost Classifier

Evaluation is primarily based on:

Macro F1-score (accounts for class imbalance)

Confusion matrices

Cross-validation grid search performance

Across models, XGBoost and Random Forest generally achieved the strongest performance, while linear models provided a useful baseline.

# Interpretability

A major component of this project is understanding why the models make the decisions they do.

Interpretability tools used include:

SHAP summary plots (global importance)

SHAP waterfall/force plots (individual predictions)

Permutation importance

XGBoost built-in importance metrics

Correlation heatmaps

UMAP visual embeddings showing class separability

The analysis highlights which financial ratios consistently drive sector predictions and assesses whether errors reflect:

inherent overlap between sectors

structural imbalance in the data

nonlinear relationships captured by tree-based models

This interpretability component fulfills the assignment requirement to make the model explainable and evaluate what was learned.

# License

This project is distributed under the terms of the MIT License, allowing users to view, modify, and extend the work with appropriate attribution.
