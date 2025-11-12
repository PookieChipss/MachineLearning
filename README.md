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

> If datasets aren’t in the repo, add the URLs below and place the CSVs beside each notebook before running.

---

## ⚙️ Environment

`ash
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
Launch notebooks:

bash
Copy code
jupyter lab   # or: jupyter notebook
🚀 Quickstart
Open Task 1/regression.ipynb or Task 2/classification.ipynb.

Run cells top→bottom (add dataset CSVs next to the notebooks if needed).

Review printed metrics and generated plots/curves.

📘 Notebook Details
Task 1 — Regression (House Prices)
Preprocessing: ColumnTransformer

Numeric → median impute (+scale where needed)

Categorical → most-frequent impute + one-hot

Target: train on log(price); back-transform predictions for reporting.

Models: RandomForestRegressor (tuned) and MLPRegressor.

Evaluation: MAE, RMSE, R² with residual checks and RF feature-importance bars.

Outcome: RF is the stronger tabular baseline; log-target stabilises error.

Task 2 — Classification (Telco Churn)
Split: stratified Train/Val/Test with robust preprocessing.

Imbalance:

RF path → ImbPipeline(preprocess → SMOTENC → RandomForest)

MLP path → dense one-hot features with class weights + EarlyStopping

Tuning: RandomizedSearchCV (StratifiedKFold, F1 scoring) for RF; sensible width/dropout grid for MLP.

Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, ROC & PR curves.

Headline (report): tuned RF+SMOTENC ≈ Accuracy ~0.77, Precision 0.55, Recall 0.68, F1 0.61, ROC-AUC 0.84, PR-AUC 0.64.
MLP tends to higher recall on churners; RF remains slightly more balanced overall.

🧩 Key Ideas
Pipelines prevent leakage (fit on Train; apply to Val/Test).

Imbalance handled correctly (SMOTENC on Train only; class weights for MLP).

Validation-driven tuning with fixed seeds (random_state=42).

Quick explainability via RF feature importances.

📊 Figures (auto-generated)
Regression: Actual vs Predicted, Residual/Error plots, RF feature importances.

Classification: Confusion Matrix, ROC & PR curves, RF feature importances.

🔗 Dataset URLs (add your exact links)
House Prices: paste source URL here

Telco Customer Churn: paste source URL here

📝 License
Academic coursework; add a formal license (e.g., MIT) if you plan to reuse.

🙌 Acknowledgements
pandas • scikit-learn • imbalanced-learn • TensorFlow/Keras • matplotlib • seaborn • Jupyter
