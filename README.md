# MachineLearning — Regression & Classification on Tabular Data
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

A compact, reproducible Semester-6 project with two end-to-end ML tasks:

- **Regression (House Prices)** — predict sale price using clean pipelines and a **log-target** strategy with clear diagnostics.  
- **Classification (Telco Churn)** — predict whether a customer will churn, using **SMOTENC** / **class weights** and tuned models.

Both tasks use **sklearn pipelines** (impute → encode/scale → model), consistent metrics, and tidy plots.

---

## 📦 Repository Structure
MachineLearning/
├─ Task 1/ # House Price Regression
│ └─ regression.ipynb
├─ Task 2/ # Telco Churn Classification
│ └─ classification.ipynb
└─ README.md

yaml
Copy code
> If datasets aren’t in the repo, use the URLs below and place the CSVs beside each notebook before running.

---

## ⚙️ Environment
```bash
# (optional) create & activate a virtual env
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# install essentials
pip install -U pip
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn jupyter

# for the Keras MLP in Task 2 (CPU TF is fine)
pip install tensorflow
```

## 🚀 Quickstart
1. Open **Task 1/regression.ipynb** or **Task 2/classification.ipynb**.
2. Run cells top to bottom (ensure the dataset CSVs are next to each notebook).
3. Review printed metrics and generated plots/curves.

## 📘 Notebook Details

### Task 1 — Regression (House Prices)
**Preprocessing:** `ColumnTransformer`  
- Numeric → median impute (+ scale where needed)  
- Categorical → most-frequent impute + one-hot

**Target:** train on `log(price)`; back-transform predictions for reporting.  
**Models:** `RandomForestRegressor` (tuned) and `MLPRegressor`.  
**Evaluation:** MAE, RMSE, R² with residual checks and RF feature-importance bars.  
**Outcome:** RF is a strong tabular baseline; log-target stabilises error.

### Task 2 — Classification (Telco Churn)
**Split:** stratified Train/Val/Test with the same preprocessing logic.  

**Imbalance handling:**  
- RF path → `ImbPipeline(preprocess → SMOTENC → RandomForest)`  
- MLP path → dense one-hot features with **class weights** + **EarlyStopping**

**Tuning:**  
- RF → `RandomizedSearchCV` (StratifiedKFold, F1 scoring)  
- MLP → small grid over width/dropout; early stopping on validation loss

**Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, ROC and PR curves.  

**Headline (report):** tuned RF + SMOTENC ≈ Accuracy ~0.77, Precision 0.55, Recall 0.68, F1 0.61, ROC-AUC 0.84, PR-AUC 0.64.  
MLP often yields higher recall on churners; RF remains more balanced overall.

## 🧩 Key Ideas
- Use pipelines to prevent leakage (fit on Train; apply to Val/Test).
- Apply SMOTENC on the training split only; use class weights for the MLP.
- Tune with validation folds and fixed seeds (`random_state=42`).
- Use RF feature importances for quick explainability.

## 📊 Figures (auto-generated)
- Regression: Actual vs Predicted, Residual/Error plots, RF feature importances.  
- Classification: Confusion Matrix, ROC & PR curves, RF feature importances.

## 📝 License
Academic coursework; add a formal license (e.g., MIT) if you plan to reuse.

## 🙌 Acknowledgements
pandas, scikit-learn, imbalanced-learn, TensorFlow/Keras, matplotlib, seaborn, Jupyter





