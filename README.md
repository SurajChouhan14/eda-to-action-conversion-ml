# eda-to-action-conversion-ml
Title: From EDA to action: conversion classification with PR-AUC–driven threshold tuning and uplift targeting, plus interactive Plotly visuals.
End-to-end marketing analytics project that predicts conversion, tunes operating thresholds for business goals (precision/recall trade-offs), and demos uplift modeling (T-Learner) to prioritize high-impact users. Includes interactive Plotly charts, leakage-safe pipelines (SMOTE inside CV), PR/ROC evaluation, feature importance, and an exportable uplift-ranked user list. Note: uplift uses simulated treatment for learning; replace with a real exposure flag to enable causal validation (Qini/AUUC).
Overview

Purpose: Predict customer conversion (0/1) from marketing data, tune the decision threshold for business goals, and demo uplift modeling to prioritize who to target.

Audience: Beginners and stakeholders. Code is step-by-step with simple comments and interactive Plotly charts.

Key deliverables: Classification pipeline (with PR-AUC focus and threshold tuning), uplift ranking (top/bottom segments), and actionable KPIs.

Open In Colab: https://colab.research.google.com/github/SurajChouhan14/eda-to-action-conversion-ml/blob/main/eda-to-action-conversion-ml.ipynb

Dataset

Source path (Kaggle): /kaggle/input/predict-conversion-in-digital-marketing-dataset/digital_marketing_campaign_dataset.csv

Target: Conversion (0/1)

Example KPIs observed:

Conversion rate ≈ 87.65%

CTR ≈ 15.48%

CPA ≈ $5,705.58

What’s inside

EDA and KPIs: Quick checks (shape, nulls), conversion rate, CTR, CPA; Plotly visuals for distributions and channel breakdowns.

Classification (primary):

Leakage-safe pipeline: ColumnTransformer (scale + one-hot) + SMOTE + XGBoost

Validation: StratifiedKFold CV with PR-AUC and ROC-AUC

Test metrics: PR-AUC ≈ 0.94, ROC-AUC ≈ 0.81 (example)

Threshold tuning: Best-F1 threshold ≈ 0.552; also supports recall-oriented thresholding

Feature importance: Global importances to guide channel/creative strategy

Uplift modeling (demo):

T-Learner (two models: P(y|treated) and P(y|control))

Treatment flag: Auto-detected if present; otherwise simulated 50/50 for learning

Results: Clear separation in demo (e.g., top 20% avg uplift ≈ +0.16; bottom 20% ≈ −0.15)

Outputs: uplift_ranked_users.csv (saved to /kaggle/working)

How to run

Open the notebook in Kaggle.

Ensure the dataset path is correct (see “Dataset” above).

Use toggles at the top:

RUN_CLASSIFICATION = True/False

RUN_UPLIFT = True/False

Run all cells. Interactive Plotly charts will render inline.

Files produced

uplift_ranked_users.csv: Ranked users by predicted uplift (saved in /kaggle/working). Download from Kaggle’s “Output/Working” panel.

Interpreting results

Classification:

Use PR-AUC to assess performance on imbalanced data.

Choose threshold based on business goals (maximize F1, or set minimum precision/recall).

Inspect confusion matrix at the chosen threshold.

Uplift:

Top uplift users are priority targets; bottom uplift users are candidates to exclude.

If treatment is simulated, treat results as a learning demo (not causal).

With a real treatment flag and randomization, validate with Qini/AUUC and an A/B holdout.

Limitations

Uplift demo uses simulated Treatment unless a real exposure column exists; simulated uplift is illustrative, not causal.

CPA/ROI depend on revenue assumptions; include AOV/CLV to compute realistic ROI.

Next steps

Replace simulated Treatment with real exposure (e.g., Exposed) and compute Qini/AUUC.

Calibrate probabilities (e.g., isotonic/Platt) and tune thresholds for cost targets (precision floor, CPA ceiling).

Hyperparameter tuning (RandomizedSearchCV/Optuna) under PR-AUC scoring.

Persist the final pipeline (preprocessing + model) and add a batch scoring function for new data.
