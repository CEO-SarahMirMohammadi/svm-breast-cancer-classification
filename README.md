# SVM Breast Cancer Classification

A machine learning project that uses a Support Vector Machine (SVM) to classify breast cancer cases using the Wisconsin Breast Cancer dataset.

The project also visualizes four important features using histograms:

- Mean Radius
- Mean Area (Size)
- Mean Texture
- Mean Smoothness

## Overview

This project demonstrates a basic supervised machine learning workflow:

1. Load the Breast Cancer dataset
2. Explore the dataset
3. Visualize selected features using histograms
4. Split the dataset into training and testing sets
5. Train a Support Vector Machine classifier
6. Make predictions on the test dataset
7. Generate a confusion matrix
8. Generate a classification report

## Algorithm

### Support Vector Machine

Support Vector Machine (SVM) is a supervised machine learning algorithm commonly used for classification problems.

The algorithm attempts to find a decision boundary that separates different classes.

In this project, the SVM model uses all available features in the Breast Cancer dataset for classification.

## Dataset

The project uses the Breast Cancer Wisconsin dataset provided by scikit-learn.

The dataset contains measurements computed from digitized images of breast mass cell nuclei.

The classification task contains two classes:

- Malignant
- Benign

The dataset contains 569 samples and 30 numerical features.

## Feature Visualization

The program creates histograms for four selected features:

### Mean Radius

Represents the average distance from the center of the nucleus to points on its perimeter.

### Mean Area

Used in this project as a measurement representing cell size.

### Mean Texture

Represents the average variation in grayscale intensity.

### Mean Smoothness

Represents the local variation in radius lengths.

## Project Structure

```text
svm-breast-cancer-classification/
│
├── svm_breast_cancer.py
├── README.md
├── requirements.txt
└── .gitignore
## Requirements

Python 3.9 or newer is recommended.

The project requires the following Python libraries:

* pandas
* numpy
* matplotlib
* scikit-learn

Install all dependencies with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
pandas
numpy
matplotlib
scikit-learn
```

## Installation

Clone the repository:

```bash
git clone https://github.com/CEO-SarahMirMohammadi/svm-breast-cancer-classification.git
```

Move into the project directory:

```bash
cd svm-breast-cancer-classification
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Program

Run the Python script:

```bash
python svm_breast_cancer.py
```

The program will:

1. Load the Breast Cancer dataset.
2. Display basic dataset information.
3. Generate histograms for radius, size, texture, and smoothness.
4. Split the dataset into training and testing sets.
5. Train an SVM classifier.
6. Predict the test data.
7. Display the confusion matrix.
8. Display the classification report.

## Machine Learning Workflow

```text
Breast Cancer Dataset
        |
        v
Feature Exploration
        |
        v
Histogram Visualization
        |
        v
Train/Test Split
        |
        v
SVM Training
        |
        v
Prediction
        |
        v
Model Evaluation
        |
        +----> Confusion Matrix
        |
        +----> Classification Report
```

## Train/Test Split

The dataset is divided into training and testing data:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=20
)
```

80% of the dataset is used for training and 20% is used for testing.

## Model Training

The SVM classifier is created using:

```python
clf = SVC()
```

The model is trained using:

```python
clf.fit(X_train, y_train)
```

## Prediction

The trained model predicts the classes of the test samples:

```python
y_predict = clf.predict(X_test)
```

## Confusion Matrix

The confusion matrix compares the predicted classes with the actual classes.

It provides information about:

* True positives
* True negatives
* False positives
* False negatives

## Classification Report

The classification report provides the following evaluation metrics:

* Precision
* Recall
* F1-score
* Support

## Example Output

The program produces output similar to:

```text
Dataset shape:
(569, 30)

Confusion Matrix:
            predicted_cancer  predicted_healthy
is_cancer                  ...
is_healthy                 ...

Classification Report:
              precision    recall  f1-score   support

           ...
```

The exact values may vary depending on the Python and scikit-learn versions.

## Key Concepts

This project demonstrates:

* Supervised learning
* Binary classification
* Support Vector Machines
* Feature exploration
* Data visualization
* Train/test splitting
* Model prediction
* Confusion matrices
* Classification metrics

## Future Improvements

Possible improvements include:

* Feature scaling using `StandardScaler`
* SVM hyperparameter tuning
* Comparing different SVM kernels
* Cross-validation
* ROC curve visualization
* Precision-recall curve
* Feature selection
* Comparing SVM with Logistic Regression
* Comparing SVM with Decision Trees
* Comparing SVM with Random Forest

## Project Requirements

| Requirement  | Version / Description     |
| ------------ | ------------------------- |
| Python       | 3.9+ recommended          |
| pandas       | Latest compatible version |
| NumPy        | Latest compatible version |
| Matplotlib   | Latest compatible version |
| scikit-learn | Latest compatible version |

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for the complete license text.

This project is intended primarily for educational and research purposes.
