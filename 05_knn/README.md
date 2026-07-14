# K-Nearest Neighbors (KNN)

## Overview

This module demonstrates the implementation of the **K-Nearest Neighbors (KNN)** classification algorithm from scratch using **NumPy**. Instead of relying on Scikit-learn's built-in `KNeighborsClassifier`, the algorithm is implemented manually to provide a deeper understanding of distance-based learning and classification.

The implementation uses the **Glass Identification Dataset** and evaluates multiple values of **k** to determine the optimal number of neighbors based on classification performance.

---

## Topics Covered

- K-Nearest Neighbors (KNN)
- Euclidean Distance
- Distance Calculation
- Majority Voting
- Hyperparameter Tuning
- Confusion Matrix
- Accuracy
- Macro Precision

---

## Files

| File | Description |
|------|-------------|
| **KNN.pdf** | Presentation explaining the KNN algorithm, mathematical concepts, and workflow. |
| **knn.py** | Complete KNN implementation from scratch using NumPy. |
| **KNN_Results.pdf** | Experimental results, performance comparison, and analysis for different values of **k**. |

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn 

---

## Machine Learning Workflow

```text
Glass Identification Dataset
           ↓
Dataset Preparation
           ↓
Train-Test Split
           ↓
Euclidean Distance Calculation
           ↓
Find k Nearest Neighbors
           ↓
Majority Voting
           ↓
Predicted Class
           ↓
Performance Evaluation
```

---

## Dataset

This implementation uses the **Glass Identification Dataset**, a publicly available dataset
**Dataset Summary**

| Attribute | Value |
|-----------|------:|
| Samples | 214 |
| Features | 9 |
| Classes | 6 |
| Train-Test Split | 70% / 30% |

---

## Experimental Setup

The classifier was evaluated using the following values of **k**:

```text
k = 1, 2, 3, 5, 10, 15
```

The following evaluation metrics were used:

- Confusion Matrix
- Accuracy
- Macro Precision

---

## Results Summary

| k | Accuracy | Macro Precision |
|---:|---------:|---------------:|
| **1** | **76.92%** | **71.64%** |
| 2 | 66.15% | 63.42% |
| 3 | 64.62% | 63.64% |
| 5 | 58.46% | 37.53% |
| 10 | 58.46% | 49.76% |
| 15 | 58.46% | 47.29% |

**Best Hyperparameter**

- **Best k:** 1
- **Highest Accuracy:** 76.92%
- **Highest Macro Precision:** 71.64%

---

## Learning Outcomes

After completing this module, you will understand how to:

- Implement the KNN algorithm from scratch.
- Compute Euclidean distances between samples.
- Identify the nearest neighbors.
- Perform majority voting for classification.
- Tune the hyperparameter **k**.
- Evaluate model performance using Accuracy and Macro Precision.

---

## Key Takeaways

- The KNN classifier was successfully implemented without using Scikit-learn's built-in KNN classifier.
- The highest classification performance was achieved with **k = 1**.
- Increasing the number of neighbors generally reduced model performance on the Glass Identification Dataset.
- This project demonstrates the complete implementation and evaluation of a distance-based machine learning algorithm.

---

