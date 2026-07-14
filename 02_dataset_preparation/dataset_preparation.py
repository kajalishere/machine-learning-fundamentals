"""
==============================================================
Module 02 : Dataset Preparation
Machine Learning Fundamentals

Author : Kajal 

Description:
This script demonstrates the fundamental steps involved in
preparing a dataset for machine learning.

Topics Covered:
1. Reading a dataset using Pandas
2. Understanding dataset dimensions
3. Identifying feature and target columns
4. Mathematical representation of the dataset
5. Train-Test Split (70:30)

GitHub:
https://github.com/kajalishere/machine-learning-fundamentals
==============================================================
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# Read Dataset
# ==========================================================

filename = "glass.csv"

df = pd.read_csv(filename)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print("Dataset Used :", filename)

# ==========================================================
# Display First Five Rows
# ==========================================================

print("\nFirst 5 Rows of Dataset")
print("-" * 60)
print(df.head())

# ==========================================================
# Dataset Shape
# ==========================================================

rows = df.shape[0]
columns = df.shape[1]

print("\nNumber of Rows :", rows)
print("Number of Columns :", columns)

# ==========================================================
# Target Column
# ==========================================================

label = df.columns[-1]

classes = sorted(df[label].unique())

print("\nNumber of Classes :", len(classes))
print("Dataset Shape :", df.shape)

# ==========================================================
# Feature Matrix (X) and Target Vector (y)
# ==========================================================

X = df.drop(label, axis=1)
y = df[label]

print("\nFeature Columns")
print(X.columns.tolist())

print("\nTarget Column")
print(label)

# ==========================================================
# Mathematical Representation
# ==========================================================

print("\n" + "=" * 60)
print("MATHEMATICAL REPRESENTATION")
print("=" * 60)

print(f"\nFeature Matrix (X) belongs to R^({rows} x {X.shape[1]})")
print(f"Target Vector (y) Classes : {classes}")

print("\nNumber of Samples (n) :", rows)
print("Number of Features (d) :", X.shape[1])

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT (70 : 30)")
print("=" * 60)

print("X_train Shape :", X_train.shape)
print("X_test Shape  :", X_test.shape)

print("y_train Shape :", y_train.shape)
print("y_test Shape  :", y_test.shape)