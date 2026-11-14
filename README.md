💳 Credit Card Fraud Detection Pipeline

An end-to-end Machine Learning pipeline in Python designed to detect fraudulent credit card transactions in a highly imbalanced dataset (where fraud accounts for only 0.17% of all records).

Rather than relying on misleading accuracy metrics, this project focuses on maximizing Fraud Recall, Precision-Recall AUC (PR-AUC), decision threshold tuning, and model explainability using SHAP.

📌 Problem Statement & Dataset

Credit card fraud detection is a classic imbalanced classification problem. In this dataset:

Normal Transactions (Class = 0): ~99.83%

Fraudulent Transactions (Class = 1): ~0.17%

A naive model that predicts "Normal" for every transaction would achieve 99.83% accuracy while failing to catch a single fraud ($Recall = 0\%$). Therefore, this pipeline shifts the evaluation focus entirely toward the minority class (Class = 1) using Recall, Precision, F1-Score, and the Precision-Recall Curve.

Dataset Details

Source: TensorFlow Official Dataset (creditcard.csv) — streamed directly via pandas.read_csv().

Features:

Time: Seconds elapsed between each transaction and the first transaction in the dataset.

V1 to V28: Principal components obtained via PCA to protect user privacy and sensitive data.

Amount: The transaction value.

Class: Target variable (1 for Fraud, 0 for Normal).

🚀 Pipeline Architecture

The script (deteccao_fraude.py) is structured into 6 core stages plus advanced evolutionary experiments:

1. Data Exploration (explorar_dados)

Streams the dataset directly from the cloud URL without requiring manual local downloads.

Inspects dataset dimensions, head records, and quantifies class imbalance using value_counts(normalize=True).

2. Preprocessing & Feature Engineering (preparar_dados)

Log Transformation: Applies $\log(1 + \text{Amount})$ via np.log1p to reduce extreme positive skewness in transaction values.

Temporal Behavior Features:

Hour_of_Day: Cyclical 24-hour representation derived from Time.

Time_Diff: Time elapsed since the immediately preceding transaction.

Stratified Split: Splits data into 80% Train and 20% Test using stratify=y to preserve the exact 0.17% fraud ratio in both sets.

Standardization: Scales features using StandardScaler fit strictly on the training set to prevent data leakage.

3. Model Training & Imbalance Handling (treinar_e_comparar_modelos)

Trains and compares multiple classification algorithms tailored for imbalanced learning:

Logistic Regression (Baseline)

Logistic Regression (class_weight='balanced')

Random Forest (class_weight='balanced')

XGBoost (scale_pos_weight dynamically tuned to negative/positive ratio)

Extra Trees (class_weight='balanced')

Resampling Comparison (comparar_reamostragem): Benchmarks Random Undersampling vs. SMOTE (Synthetic Minority Over-sampling Technique) on Recall performance.

4. Evaluation Beyond Accuracy (avaliar_modelo & plotar_curvas_avaliacao)

Generates detailed Classification Reports and Confusion Matrices.

Computes ROC-AUC and PR-AUC (Average Precision).

Exports side-by-side plots of the ROC Curve and Precision-Recall Curve (curvas_roc_e_precision_recall.png).

5. Pipeline Integration, Threshold Tuning & GridSearchCV (ajustar_pipeline_e_limiar)

Wraps StandardScaler and RandomForestClassifier into a single scikit-learn Pipeline to guarantee zero data leakage during cross-validation.

Runs GridSearchCV with StratifiedKFold, optimizing hyperparameters specifically for scoring='recall'.

Decision Threshold Tuning: Lowers the classification probability threshold from 0.5 to 0.3 to capture high-risk borderline transactions and boost Fraud Recall.

6. Model Explainability (explicar_modelo)

Global Feature Importance: Extracts and plots the top 12 most influential features (importancia_variaveis.png).

SHAP (SHapley Additive exPlanations):

Generates a global SHAP Summary Plot (shap_summary_plot.png) showing how feature values impact fraud probability.

Provides Local Explainability for individual fraudulent transactions, detailing the exact top 5 features that pushed the model to flag a specific transaction as fraud.

🛠️ Installation & Requirements

Clone the repository and install the required Python packages:

pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn shap


▶️ How to Run

Execute the Python script directly from your terminal or run it cell-by-cell (# %%) in VS Code / Spyder / Jupyter:

python deteccao_fraude.py


📊 Generated Outputs

Running the script outputs a comprehensive console report (including a final leaderboard sorted by Fraud Recall) and automatically saves three high-resolution charts in the project directory:

curvas_roc_e_precision_recall.png — ROC and Precision-Recall curves across all models.

importancia_variaveis.png — Top 12 most important features from the Random Forest model.

shap_summary_plot.png — SHAP value distribution showing feature impact on fraud predictions.

🧠 Key Takeaways

Accuracy is a vanity metric in fraud detection; Recall and PR-AUC are the primary drivers of business value.

Lowering the decision threshold (from 0.5 to 0.3) is a straightforward, high-impact way to catch more fraudulent transactions without retraining the model.

Explainable AI (SHAP) bridges the gap between complex ensemble models and auditing/compliance teams by explaining why a specific transaction was blocked.
