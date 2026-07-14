"""
==============================================================
Module 05 : K-Nearest Neighbors (KNN)
Machine Learning Fundamentals

Author : Kajal 

Description:
This script implements the K-Nearest Neighbors (KNN) algorithm
from scratch using NumPy without relying on Scikit-learn's
KNeighborsClassifier.

Topics Covered:
1. Euclidean Distance
2. KNN Classification
3. Majority Voting
4. Confusion Matrix
5. Accuracy
6. Precision
7. Hyperparameter Tuning (k)

Dataset:
Glass Identification Dataset

GitHub:
https://github.com/kajalishere/machine-learning-fundamentals
==============================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score


filename = "glass.csv"
df = pd.read_csv(filename)
print("Dataset used      :", filename)
print("Columns           :", list(df.columns))   # verify which column is the label
print(df.head(), "\n")

label = df.columns[-1]                      # convention: last column = class label
X = df.drop(label, axis=1).to_numpy()       # features  -> NumPy array
y = df[label].to_numpy()                    # labels    -> NumPy array

n, d = X.shape
print("n (rows/examples) :", n)
print("d (features)      :", d, "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
print("X_train shape     :", X_train.shape)
print("X_test  shape     :", X_test.shape, "\n")

n_train = X_train.shape[0]   # theoretical max value k could take is n_train

# ----------------------------------------------------------------------
# kNN FROM SCRATCH
# ----------------------------------------------------------------------
def predict_one(x_query, X_train, y_train, k):
    """Predict the label of a single point x_query."""
    # Euclidean distance from x_query to EVERY training point.
    # (X_train - x_query) subtracts x_query from each row (broadcasting),
    # we square, sum across the feature axis, then take the square root.
    distances = np.sqrt(np.sum((X_train - x_query) ** 2, axis=1))

    # Indices of the k SMALLEST distances (the k nearest neighbors).
    #returns the indices that would sort an array of distances in ascending order (closest to farthest)
    k_idx = np.argsort(distances)[:k]

    # The labels of those k neighbors.
    k_labels = y_train[k_idx]

    # Majority vote: the label that appears most often wins.
    # np.unique returns sorted labels, so ties break toward the smaller label.
    values, counts = np.unique(k_labels, return_counts=True)
    return values[np.argmax(counts)]


def knn_predict(X_query, X_train, y_train, k):
    """Predict labels for every point in X_query."""
    return np.array([predict_one(x, X_train, y_train, k) for x in X_query])


# ----------------------------------------------------------------------
# 5. TEST DIFFERENT k VALUES
# ----------------------------------------------------------------------
k_values = [1,2, 3, 5, 10, 15]   # k can range from 1 up to n_train; these are the ones we test

results = []   # store (k, accuracy, precision) for the final comparison

for k in k_values:
    print("=" * 55) # prints * 55 times
    print(f"k = {k}") #print which k value
    print("=" * 55)

    # Predict on the test set
    y_pred = knn_predict(X_test, X_train, y_train, k)

    # Confusion matrix: rows = actual, columns = predicted
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(cm)

    # Accuracy = fraction of correct predictions
    acc = accuracy_score(y_test, y_pred)

    # Precision (macro = average of per-class precision; works for any labels)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"Accuracy          : {acc:.4f}")
    print(f"Precision (macro) : {prec:.4f}\n")

    results.append((k, acc, prec))

# ----------------------------------------------------------------------
# 6. FIND THE BEST k FOR ACCURACY AND FOR PRECISION
# ----------------------------------------------------------------------
best_acc  = max(results, key=lambda r: r[1])   # r[1] = accuracy
best_prec = max(results, key=lambda r: r[2])   # r[2] = precision

print("=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"{'k':>4} | {'accuracy':>9} | {'precision':>9}") #use for alignment
print("-" * 30)
for k, acc, prec in results:
    print(f"{k:>4} | {acc:>9.4f} | {prec:>9.4f}")

print("-" * 30)
print(f"Best accuracy : k = {best_acc[0]}  (accuracy  = {best_acc[1]:.4f})")
print(f"Best precision: k = {best_prec[0]}  (precision = {best_prec[2]:.4f})")

# ----------------------------------------------------------------------
# 7. VISUALIZE RESULTS
# ----------------------------------------------------------------------

import matplotlib.pyplot as plt

# Extract values from results
k_list = [r[0] for r in results]
accuracy_list = [r[1] for r in results]
precision_list = [r[2] for r in results]

# -------------------------------
# Accuracy vs K
# -------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    k_list,
    accuracy_list,
    marker='o',
    linewidth=2,
    label='Accuracy'
)

plt.title("Accuracy vs k")

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")

plt.xticks(k_list)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig("accuracy_vs_k.png", dpi=300)

plt.show()

# -------------------------------
# Precision vs K
# -------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    k_list,
    precision_list,
    marker='s',
    linewidth=2,
    label='Macro Precision'
)

plt.title("Precision vs k")

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Macro Precision")

plt.xticks(k_list)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig("precision_vs_k.png", dpi=300)

plt.show()