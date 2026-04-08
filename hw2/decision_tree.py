from typing import Self

import numpy as np


class Node:

    def __init__(self, feature_index: int, threshold: float, left: Self | None,
                 right: Self | None, label: int):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label


class DecisionTree:

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 sample_weights: np.ndarray | None = None,
                 max_depth: int = 5,
                 minimum_samples: int = 3,
                 gain_function: str = "gini"):
        self.x = x
        self.y = y.astype(np.int64, copy=False)
        if sample_weights is None:
            self.sample_weights = np.ones(self.y.shape[0], dtype=float)
        else:
            if sample_weights.shape[0] != self.y.shape[0]:
                raise ValueError(
                    "sample_weights must have the same length as y")
            self.sample_weights = sample_weights.astype(float, copy=False)
        self.max_depth = max_depth
        self.minimum_samples = minimum_samples
        self.gain_function = gain_function

        self.root = self.build_tree(x,
                                    self.y,
                                    self.sample_weights,
                                    max_depth=max_depth)

    def build_tree(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                   max_depth: int) -> Node:
        label = self.weighted_majority_label(y, weights)

        if max_depth == 0 or len(y) < self.minimum_samples or len(
                np.unique(y)) == 1:
            return Node(-1, float("nan"), None, None, label)

        feature_index, threshold = self.find_best_split(x, y, weights)
        if feature_index < 0:
            return Node(-1, float("nan"), None, None, label)

        left_indices = x[:, feature_index] <= threshold
        right_indices = ~left_indices
        if left_indices.sum() == 0 or right_indices.sum() == 0:
            return Node(-1, float("nan"), None, None, label)

        left_x, left_y = x[left_indices], y[left_indices]
        right_x, right_y = x[right_indices], y[right_indices]
        left_w, right_w = weights[left_indices], weights[right_indices]

        if len(left_y) < self.minimum_samples or len(
                right_y) < self.minimum_samples:
            return Node(-1, float("nan"), None, None, label)

        left_node = self.build_tree(left_x, left_y, left_w, max_depth - 1)
        right_node = self.build_tree(right_x, right_y, right_w, max_depth - 1)
        return Node(feature_index, threshold, left_node, right_node, label)

    def find_best_split(self, x: np.ndarray, y: np.ndarray,
                        weights: np.ndarray) -> tuple[int, float]:
        best_gain = -np.inf
        best_feature_index = -1
        best_threshold = -1
        for feature_index in range(x.shape[1]):
            nums = np.unique(x[:, feature_index])
            if nums.size < 2:
                continue
            thresholds = (nums[:-1] + nums[1:]) / 2
            for threshold in thresholds:
                left_indices = x[:, feature_index] <= threshold
                right_indices = x[:, feature_index] > threshold
                if left_indices.sum() == 0 or right_indices.sum() == 0:
                    continue
                left_y, right_y = y[left_indices], y[right_indices]
                left_w, right_w = weights[left_indices], weights[right_indices]
                gain = self.calculate_gain(y, weights, left_y, left_w, right_y,
                                           right_w)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        return best_feature_index, best_threshold

    def calculate_gain(self, parent_y: np.ndarray, parent_w: np.ndarray,
                       left_y: np.ndarray, left_w: np.ndarray,
                       right_y: np.ndarray, right_w: np.ndarray) -> float:
        parent_weight = float(np.sum(parent_w))
        left_weight = float(np.sum(left_w))
        right_weight = float(np.sum(right_w))
        if parent_weight <= 0 or left_weight <= 0 or right_weight <= 0:
            return -np.inf

        if self.gain_function == "gini":
            impurity = self.gini
        else:
            impurity = self.entropy

        return impurity(
            parent_y, parent_w) - (left_weight / parent_weight) * impurity(
                left_y, left_w) - (right_weight / parent_weight) * impurity(
                    right_y, right_w)

    def weighted_majority_label(self, y: np.ndarray,
                                weights: np.ndarray) -> int:
        if len(y) == 0:
            return 0
        classes, inverse = np.unique(y, return_inverse=True)
        class_weight = np.bincount(inverse, weights=weights)
        return int(classes[int(np.argmax(class_weight))])

    def gini(self, y: np.ndarray, weights: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, inverse = np.unique(y, return_inverse=True)
        class_weight = np.bincount(inverse, weights=weights)
        total_weight = float(np.sum(class_weight))
        if total_weight <= 0:
            return 0.0
        proportions = class_weight / total_weight
        return 1 - np.sum(proportions**2)

    def entropy(self, y: np.ndarray, weights: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, inverse = np.unique(y, return_inverse=True)
        class_weight = np.bincount(inverse, weights=weights)
        total_weight = float(np.sum(class_weight))
        if total_weight <= 0:
            return 0.0
        proportions = class_weight / total_weight
        return -np.sum(proportions * np.log2(proportions + 1e-10))

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = []
        for sample in x:
            node = self.root
            while node.left is not None and node.right is not None:
                if sample[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.label)
        return np.array(predictions)


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
    tree = DecisionTree(train_x, train_y, max_depth=5, minimum_samples=3)
    predictions = tree.predict(test_x)
    accuracy, precision, recall, f1_score = evaluate(test_y, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    predictions = tree.predict(train_x)
    accuracy, precision, recall, f1_score = evaluate(train_y, predictions)
    print(f"Train Accuracy: {accuracy:.4f}")
    print(f"Train Precision: {precision:.4f}")
    print(f"Train Recall: {recall:.4f}")
    print(f"Train F1 Score: {f1_score:.4f}")
