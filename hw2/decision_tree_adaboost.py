# decision_tree_adaboost

import numpy as np

from decision_tree import DecisionTree, Node, evaluate


class AdaBoost:

    def __init__(self, x: np.ndarray, y: np.ndarray, n_estimators: int = 50):
        self.x = x
        # Internally use {-1, +1} labels for standard AdaBoost update.
        self.y = np.where(y == 0, -1, 1).astype(np.int64, copy=False)
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []

    def fit(self):
        n_samples = self.x.shape[0]
        sample_weights = np.ones(n_samples) / n_samples

        y_pm = self.y

        for _ in range(self.n_estimators):
            estimator = DecisionTree(self.x,
                                     y_pm,
                                     sample_weights=sample_weights,
                                     max_depth=2,
                                     minimum_samples=2)

            pred_pm = estimator.predict(self.x).astype(np.int64, copy=False)
            pred_pm = np.where(pred_pm == 0, -1, pred_pm)

            incorrect = pred_pm != y_pm
            error = float(np.sum(sample_weights[incorrect]))
            error = float(np.clip(error, 1e-12, 1 - 1e-12))

            # If the weak learner is not better than random guessing, stop.
            if error >= 0.5:
                break

            estimator_weight = 0.5 * np.log((1 - error) / error)

            sample_weights *= np.exp(-estimator_weight * y_pm * pred_pm)
            sample_weights /= np.sum(sample_weights)

            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)

    def predict(self, x: np.ndarray) -> np.ndarray:
        final_scores = np.zeros(x.shape[0], dtype=float)
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            pred_pm = estimator.predict(x).astype(np.int64, copy=False)
            pred_pm = np.where(pred_pm == 0, -1, pred_pm)
            final_scores += weight * pred_pm
        return (final_scores > 0).astype(np.int64)


if __name__ == '__main__':
    data = np.load("moons_3d.npz")
    train_x, train_y = data["train_x"], data["train_y"].astype(np.int64)
    test_x, test_y = data["test_x"], data["test_y"].astype(np.int64)

    tree = AdaBoost(train_x, train_y, n_estimators=50)
    tree.fit()
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
