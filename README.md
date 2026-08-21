# Breast Cancer Classification — SLP → MLP → Regularization

## 📌 Project Overview

This project focuses on **Breast Cancer Classification** using the **Wisconsin Diagnostic Breast Cancer dataset** and Deep Learning models built with **TensorFlow/Keras**.

The project starts with a **Single-Layer Perceptron (SLP)** as a linear baseline and gradually improves the neural network using **Multi-Layer Perceptron (MLP)**, different activation functions, **Early Stopping**, **Dropout**, and **L2 Regularization**.

## 📂 Dataset

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

* Samples: **569**
* Features: **30**
* Target Classes:

  * `0` = Malignant
  * `1` = Benign
* Malignant: **212**
* Benign: **357**

The dataset has a mild class imbalance, so the project uses a stratified train-test split and evaluates Precision, Recall and F1-score.

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## 🔄 Project Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Class Distribution Analysis
4. Feature Correlation Analysis
5. Train-Test Split
6. Standard Scaling
7. Single-Layer Perceptron
8. Multi-Layer Perceptron
9. Activation Function Comparison
10. Early Stopping
11. Dropout
12. L1/L2 Regularization
13. Final Model
14. Model Comparison

The notebook uses `StandardScaler`, `train_test_split`, classification metrics and TensorFlow/Keras for model development.

## 🧠 Models

### 1. Single-Layer Perceptron

A single neuron with sigmoid activation is used as the baseline model. It is equivalent to a linear/logistic classification approach.

### 2. Multi-Layer Perceptron

A `64 → 32 → 1` neural network is trained with different hidden-layer activation functions including:

* ReLU
* Tanh
* Sigmoid

### 3. Early Stopping

A larger `128 → 64 → 1` network is trained with Early Stopping to stop training when validation performance stops improving.

### 4. Dropout

Different dropout rates are compared to improve generalization and reduce overfitting.

### 5. L2 Regularization

L2 regularization with a value of `0.001` is tested to control model complexity.

### 6. Final Combined Model

The final architecture combines:

* Dense layers: `128 → 64 → 1`
* ReLU hidden activations
* L2 regularization `0.001`
* Dropout `0.3`
* Early Stopping
* Adam optimizer
* Binary Cross-Entropy loss

The final model achieved **95.61% accuracy, 98.55% precision, 94.44% recall and 96.45% F1-score** on the test set.

## 📊 Results

| Model                |   Accuracy |  Precision |     Recall |         F1 |
| -------------------- | ---------: | ---------: | ---------: | ---------: |
| SLP                  |     92.11% |     95.65% |     91.67% |     93.62% |
| MLP-ReLU             |     96.49% |     98.57% |     95.83% |     97.18% |
| MLP + Early Stopping |     96.49% |     98.57% |     95.83% |     97.18% |
| MLP + Dropout        |     96.49% |     98.57% |     95.83% |     97.18% |
| MLP + L2             |     93.86% |     98.51% |     91.67% |     94.96% |
| **Final Combined**   | **95.61%** | **98.55%** | **94.44%** | **96.45%** |

Results are taken from the notebook's generated comparison table.

## 🏆 Conclusion

The project demonstrates how a basic SLP can be improved using deeper neural networks and regularization techniques.

The **MLP-ReLU** configuration produced the highest test Accuracy and F1-score in the comparison table, while the **Final Combined Model** was selected as the preferred deployment candidate because it combines Dropout, L2 Regularization and Early Stopping for better generalization.

## 📁 Project Outputs

The notebook generates:

* EDA plots
* Activation comparison plots
* Early Stopping comparison
* Dropout comparison
* Regularization comparison
* `results_comparison_table.csv`

Plots are saved in the `/plots` directory.

## ⚠️ Note

This is a machine-learning research/educational project. The model should support, not replace, clinical decision-making. The notebook also notes that because false negatives can be particularly important in this classification task, threshold selection should be considered carefully.
