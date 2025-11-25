# Injury Risk Prediction — University Football Dataset

This project builds a machine learning pipeline to predict the probability that a university football athlete will suffer an injury in the next season. The workflow includes exploratory data analysis, preprocessing, feature engineering, model training, evaluation, and SHAP-based interpretability.

## Dataset

Source: University Football Injury Prediction Dataset (Kaggle).  
Local copy used: `data/data.csv`.

If the dataset cannot be included due to licensing, download it from Kaggle and place it manually at:
 https://www.kaggle.com/datasets/yuanchunhong/university-football-injury-prediction-dataset?resource=download

## Notebooks Overview

### 01_exploration.ipynb
- Loads dataset.
- Checks shape, dtypes, missing values, outliers.
- Examines target distribution.
- Generates histograms, boxplots, and correlation matrix.

### 02_preprocessing_feature_engineering.ipynb
- Cleans data and normalizes categorical variables.
- Builds ColumnTransformer with numeric and categorical pipelines.
- Creates engineered features:
  - Readiness_Strength
  - Workload_Index
  - Prep_Score
  - Injury_History_Weight
- Saves preprocessing artifacts into `artifacts/`:
  - preprocessor.pkl
  - numeric_features.pkl
  - categorical_features.pkl

### 03_model_training.ipynb
- Trains Logistic Regression, Random Forest, and Gradient Boosting.
- Uses recall as the priority metric.
- Computes accuracy, precision, recall, F1, and ROC AUC.
- Saves final model pipeline to:
  artifacts/random_forest_model.pkl

### 04_interpretation_conclusions.ipynb
- Loads saved pipeline from `artifacts/`.
- Computes confusion matrix and full evaluation.
- Generates SHAP summary, dependence, and waterfall plots.
- Saves all plots and tables to `artifacts/`.

## Artifacts Location

All generated outputs are now stored inside:

artifacts/

This includes:
- random_forest_model.pkl  
- feature_importance_rf.csv  
- shap_feature_importance_rf.csv  
- confusion_matrix.png  
- shap_summary.png  
- shap_dependence.png  
- shap_waterfall.png  

Only the raw input dataset remains in:

data/data.csv

## How to Run

1) Create virtual environment:

python -m venv .venv  
source .venv/bin/activate      (macOS/Linux)  
.venv\Scripts\activate         (Windows)

2) Install dependencies:

pip install -r requirements.txt

3) Run notebooks in order:

01 → 02 → 03 → 04

If `artifacts/random_forest_model.pkl` already exists, you may start directly at notebook 04.

## About src/__init__.py

Marks the `src/` directory as a Python package.  
Not required by the notebooks, but keeps the repository organized.

Minimal content:

"""src package for the Injury Prediction project."""  
__all__ = []

## Requirements

pandas  
numpy  
scikit-learn  
matplotlib  
seaborn  
joblib  
shap  
xgboost

## Summary

This project provides a full injury prediction pipeline with engineered features, trained models, evaluation, and SHAP interpretability. It is designed for study and portfolio purposes, demonstrating a clean end-to-end machine learning workflow.


## Future Improvements

This project was designed as a study-oriented end-to-end machine learning pipeline.  
Several enhancements could be implemented in a future version to increase robustness, scalability, and industry alignment:

### 1. Integrate Feature Engineering into the Pipeline
Currently, engineered features (Readiness_Strength, Workload_Index, Prep_Score, Injury_History_Weight) are created manually in the notebooks.  
A future version could include a custom `FeatureEngineeringTransformer` inside the sklearn `Pipeline` to ensure:

- automatic reproducibility,
- simplified inference workflows,
- cleaner notebook logic,
- consistent preprocessing across training and deployment.

### 2. Add Model Versioning and Experiment Tracking
Tools such as MLflow, Weights & Biases, or DVC could be used to track:

- hyperparameters,
- datasets,
- performance metrics,
- experiment comparisons.

This would make experimentation more transparent and reproducible.

### 3. Improve Outlier Detection and Data Validation
Implementing robust checks before training could prevent data quality issues:

- schema validation (with Pandera or pydantic)
- automated outlier detection (IQR, Isolation Forest)
- data drift checks for future datasets

### 4. Support Probabilistic Predictions and Threshold Tuning
Instead of returning only binary predictions (0 or 1), future versions could expose injury risk as a continuous probability using `predict_proba`. This allows more nuanced decision-making and enables the use of custom risk thresholds aligned with performance or medical objectives.

- probability outputs (`predict_proba[:, 1]`)  
- adjustable decision thresholds (e.g., prioritizing recall for injury prevention)  
- calibrated probabilities using `CalibratedClassifierCV` (sigmoid or isotonic)  
- risk banding (e.g., low / medium / high risk groups)

## Final Report
The full report of the project can be accessed here:

📄 **[Final Report (PDF)](docs/final_report.pdf)**
