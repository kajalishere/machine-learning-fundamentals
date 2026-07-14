
"""
Created on Sun Jun 28 21:27:03 2026

@author: kajal


Data Preprocessing Demo
- Check missing values
- Handle missing values using imputation
- Detect outliers using IQR and Z-score
"""

import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------
filename = "glass.csv"   
df = pd.read_csv(filename)

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

print("Dataset used:", filename)
print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. CHECK MISSING VALUES
# --------------------------------------------------
print("\nMissing values in each column:")
print(df.isnull().sum())

total_missing = df.isnull().sum().sum()
print("\nTotal missing values:", total_missing)

# --------------------------------------------------
# 3. HANDLE MISSING VALUES USING IMPUTATION
# --------------------------------------------------
# Separate numerical columns
numeric_cols = df.select_dtypes(include=np.number).columns

# Fill missing numerical values with column mean
df_imputed = df.copy()
df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(
    df_imputed[numeric_cols].mean()
)

print("\nMissing values after mean imputation:")
print(df_imputed.isnull().sum())

# --------------------------------------------------
# 4. DETECT OUTLIERS USING IQR METHOD
# --------------------------------------------------
print("\n" + "=" * 60)
print("OUTLIER DETECTION USING IQR METHOD")
print("=" * 60)

# Exclude target/class column
label = df.columns[-1]
feature_cols = df.drop(label, axis=1).select_dtypes(include=np.number).columns

for col in feature_cols:
    Q1 = df_imputed[col].quantile(0.25)
    Q3 = df_imputed[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df_imputed[
        (df_imputed[col] < lower_limit) |
        (df_imputed[col] > upper_limit)
    ]

    print(f"\nColumn: {col}")
    print("Lower Limit:", round(lower_limit, 3))
    print("Upper Limit:", round(upper_limit, 3))
    print("Number of Outliers:", outliers.shape[0])

# --------------------------------------------------
# 5. DETECT OUTLIERS USING Z-SCORE METHOD
# --------------------------------------------------
print("\n" + "=" * 60)
print("OUTLIER DETECTION USING Z-SCORE METHOD")
print("=" * 60)

for col in feature_cols:
    mean = df_imputed[col].mean()
    std = df_imputed[col].std()

    z_scores = (df_imputed[col] - mean) / std

    outliers = df_imputed[abs(z_scores) > 3]

    print(f"\nColumn: {col}")
    print("Number of Outliers:", outliers.shape[0])