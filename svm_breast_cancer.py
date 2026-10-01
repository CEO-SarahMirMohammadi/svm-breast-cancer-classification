# SVM Breast Cancer Classification
# Repository #2
#
# This program:
# 1. Loads the Breast Cancer dataset
# 2. Plots histograms of radius, size, texture, and smoothness
# 3. Splits the data into training and testing sets
# 4. Trains an SVM classifier
# 5. Makes predictions
# 6. Displays the confusion matrix and classification report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------
# 1. Load the Breast Cancer dataset
# ---------------------------------------------------------

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target


# ---------------------------------------------------------
# 2. Create a DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)

print("Dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nFeature names:")
print(cancer.feature_names)


# ---------------------------------------------------------
# 3. Plot histograms
# ---------------------------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(df["mean radius"], bins=20)
plt.xlabel("Mean Radius")
plt.ylabel("Frequency")
plt.title("Histogram of Mean Radius")

plt.subplot(2, 2, 2)
plt.hist(df["mean area"], bins=20)
plt.xlabel("Mean Area (Size)")
plt.ylabel("Frequency")
plt.title("Histogram of Mean Size")

plt.subplot(2, 2, 3)
plt.hist(df["mean texture"], bins=20)
plt.xlabel("Mean Texture")
plt.ylabel("Frequency")
plt.title("Histogram of Mean Texture")

plt.subplot(2, 2, 4)
plt.hist(df["mean smoothness"], bins=20)
plt.xlabel("Mean Smoothness")
plt.ylabel("Frequency")
plt.title("Histogram of Mean Smoothness")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 4. Split the dataset
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=20
)


# ---------------------------------------------------------
# 5. Create and train the SVM classifier
# ---------------------------------------------------------

clf = SVC()

clf.fit(X_train, y_train)


# ---------------------------------------------------------
# 6. Make predictions
# ---------------------------------------------------------

y_predict = clf.predict(X_test)


# ---------------------------------------------------------
# 7. Create the confusion matrix
# ---------------------------------------------------------

cm = np.array(
    confusion_matrix(
        y_test,
        y_predict,
        labels=[1, 0]
    )
)

confusion = pd.DataFrame(
    cm,
    index=["is_cancer", "is_healthy"],
    columns=["predicted_cancer", "predicted_healthy"]
)

print("\nConfusion Matrix:")
print(confusion)


# ---------------------------------------------------------
# 8. Classification report
# ---------------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_predict))