"""
==============================================================
Module 04 : Model Evaluation
Machine Learning Fundamentals

Author : Kajal 

Description:
This script demonstrates how to evaluate a binary classification
model using the Glass Identification Dataset.

Topics Covered
--------------
1. Confusion Matrix
2. Accuracy
3. Precision
4. Recall
5. F1-Score
6. True Positive Rate (TPR)
7. True Negative Rate (TNR)
8. False Positive Rate (FPR)
9. False Negative Rate (FNR)
10. ROC Curve
11. AUC Score

GitHub:
https://github.com/kajalishere/machine-learning-fundamentals
==============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# ==========================================================
# LOAD DATASET
# ==========================================================

filename = "glass.csv"

df = pd.read_csv(filename)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("Dataset :", filename)
print("Shape   :", df.shape)

# ==========================================================
# FEATURES AND TARGET
# ==========================================================

label = df.columns[-1]

X = df.drop(label, axis=1)
y = df[label]

# ==========================================================
# CONVERT MULTI-CLASS TO BINARY
#
# Class 1 = Positive
# All Other Classes = Negative
# ==========================================================

y = (y == 1).astype(int)

print("\nBinary Class Distribution")

print(y.value_counts())

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# TRAIN LOGISTIC REGRESSION MODEL
# ==========================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)

print(cm)

TN, FP, FN, TP = cm.ravel()

print("\nTrue Positive :", TP)
print("True Negative :", TN)
print("False Positive:", FP)
print("False Negative:", FN)

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall   : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

# ==========================================================
# DERIVED METRICS
# ==========================================================

TPR = TP / (TP + FN)

TNR = TN / (TN + FP)

FPR = FP / (FP + TN)

FNR = FN / (FN + TP)

print("\n" + "=" * 60)
print("DERIVED METRICS")
print("=" * 60)

print(f"True Positive Rate (TPR): {TPR:.4f}")

print(f"True Negative Rate (TNR): {TNR:.4f}")

print(f"False Positive Rate(FPR): {FPR:.4f}")

print(f"False Negative Rate(FNR): {FNR:.4f}")

# ==========================================================
# ROC CURVE
# ==========================================================

fpr, tpr, threshold = roc_curve(
    y_test,
    y_prob
)

auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n" + "=" * 60)
print("ROC CURVE")
print("=" * 60)

print(f"AUC Score : {auc:.4f}")

# ==========================================================
# PLOT ROC
# ==========================================================

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"ROC Curve (AUC = {auc:.2f})"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    color="red",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("Receiver Operating Characteristic (ROC) Curve")

plt.legend()

plt.grid(True)

plt.show()

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall   : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

print(f"AUC      : {auc:.4f}")

print("\nModel evaluation completed successfully.")