# 🏥 Injury Risk Prediction in Football  
**Probabilistic injury risk modeling for performance and prevention**

## Context
Injuries are one of the most critical challenges in football, affecting player availability, squad planning, and long-term performance.  
Instead of attempting to *predict injuries deterministically*, this project focuses on **estimating injury risk probabilities** to support preventive and monitoring decisions.

This is a **study-oriented, end-to-end machine learning project**, designed to simulate how injury risk modeling could be approached in a football analytics context.

---

## Objective
Estimate the **probability that a football athlete will suffer an injury in the following season**, using historical workload, readiness, and injury-related variables.

The model is intended as a **decision-support tool**, not a medical diagnosis or definitive predictor.

---

## Dataset
- **Source:** University Football Injury Prediction Dataset (Kaggle)  
- **Level:** University athletes  
- **Key limitation:**  
  The dataset does not represent professional football environments. Results should be interpreted as a **methodological demonstration**, not a production-ready solution.

Dataset link:  
https://www.kaggle.com/datasets/yuanchunhong/university-football-injury-prediction-dataset

---

## Methodology
The project follows a complete machine learning workflow:

- **Exploratory Data Analysis**
  - Target distribution, missing values, outliers, correlations

- **Preprocessing & Feature Engineering**
  - Separate numeric and categorical pipelines using `ColumnTransformer`
  - Engineered features inspired by football performance concepts:
    - `Readiness_Strength`
    - `Workload_Index`
    - `Prep_Score`
    - `Injury_History_Weight`

- **Model Training**
  - Logistic Regression  
  - Random Forest  
  - Gradient Boosting  
  - **Recall prioritized** to reduce missed injury-risk cases

- **Evaluation & Interpretability**
  - Accuracy, Precision, Recall, F1-score, ROC AUC
  - SHAP analysis for global and individual-level interpretability

---

## Key Findings
- Tree-based models outperformed linear baselines.
- Injury history and workload-related features showed the highest influence on risk estimation.
- SHAP visualizations enabled transparent interpretation of model outputs.

---

## How This Could Be Used in a Football Club
In a professional environment, a similar approach could be used to:
- identify players with **elevated injury risk**,
- support **load management and rotation decisions**,
- complement medical and performance staff assessments,
- group players into **risk bands** (low / medium / high) instead of binary labels.

---

## Repository Structure
- `data/` — raw dataset  
- `notebooks/` — step-by-step analysis and modeling  
- `artifacts/` — saved models, preprocessing objects, evaluation outputs  

---

## How to Run

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate  # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks in the following order:

1. `01_exploration.ipynb`  
2. `02_preprocessing_feature_engineering.ipynb`  
3. `03_model_training.ipynb`  
4. `04_interpretation_conclusions.ipynb`

---

## Future Improvements
- Integrate feature engineering directly into the sklearn Pipeline  
- Add experiment tracking (MLflow / W&B / DVC)  
- Improve data validation and outlier detection  
- Expose calibrated probabilistic outputs and adjustable decision thresholds  
- Apply the methodology to professional-level datasets  

---

## Final Report
📄 Full project report available here:  
📄 **[Final Report (PDF)](docs/final_report.pdf)**
