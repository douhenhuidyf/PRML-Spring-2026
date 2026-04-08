from typing import Self

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Support Vector Machines (SVM) with at least three different kernel functions. T


def evaluate(y_true: np.ndarray,
             y_pred: np.ndarray) -> tuple[float, float, float, float]:
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    f1_score = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1_score


if __name__ == "__main__":
    data = np.load("moons_3d.npz")
    train_x, train_y = data["train_x"], np.astype(data["train_y"], np.int64)
    test_x, test_y = data["test_x"], np.astype(data["test_y"], np.int64)

    kernels = ['linear', 'poly', 'rbf']
    for kernel in kernels:
        model = SVC(kernel=kernel)
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        accuracy, precision, recall, f1_score = evaluate(test_y, predictions)
        print(f"Kernel: {kernel}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}\n")