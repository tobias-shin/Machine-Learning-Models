from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import inf
from typing import Any


@dataclass
class Node:
    prediction: Any
    feature_index: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    impurity: float = 0.0
    samples: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class DecisionTreeClassifier:
    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Node | None = None
        self.n_features_: int = 0

    def fit(self, X: list[list[float]], y: list[Any]) -> "DecisionTreeClassifier":
        if not X or not y:
            raise ValueError("X and y must be non-empty.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        self.n_features_ = len(X[0])
        if self.n_features_ == 0:
            raise ValueError("X must contain at least one feature.")
        if any(len(row) != self.n_features_ for row in X):
            raise ValueError("All rows in X must have the same number of features.")

        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: list[list[float]]) -> list[Any]:
        if self.root is None:
            raise ValueError("The tree has not been fit yet.")
        return [self._predict_row(row, self.root) for row in X]

    def score(self, X: list[list[float]], y: list[Any]) -> float:
        predictions = self.predict(X)
        correct = sum(pred == actual for pred, actual in zip(predictions, y))
        return correct / len(y)

    def print_tree(self) -> None:
        if self.root is None:
            raise ValueError("The tree has not been fit yet.")
        self._print_node(self.root)

    def _build_tree(self, X: list[list[float]], y: list[Any], depth: int) -> Node:
        prediction = self._majority_class(y)
        impurity = self._gini(y)
        node = Node(prediction=prediction, impurity=impurity, samples=len(y))

        if impurity == 0.0:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(y) < self.min_samples_split:
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature_index, threshold, left_indices, right_indices = split
        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(left_X, left_y, depth + 1)
        node.right = self._build_tree(right_X, right_y, depth + 1)
        return node

    def _best_split(
        self, X: list[list[float]], y: list[Any]
    ) -> tuple[int, float, list[int], list[int]] | None:
        best_gain = -inf
        best_split: tuple[int, float, list[int], list[int]] | None = None
        parent_impurity = self._gini(y)

        for feature_index in range(self.n_features_):
            values = sorted({row[feature_index] for row in X})
            if len(values) < 2:
                continue

            thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
            for threshold in thresholds:
                left_indices = [i for i, row in enumerate(X) if row[feature_index] <= threshold]
                right_indices = [i for i, row in enumerate(X) if row[feature_index] > threshold]

                if not left_indices or not right_indices:
                    continue

                gain = self._information_gain(y, left_indices, right_indices, parent_impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold, left_indices, right_indices)

        if best_gain <= 0:
            return None
        return best_split

    def _information_gain(
        self,
        y: list[Any],
        left_indices: list[int],
        right_indices: list[int],
        parent_impurity: float,
    ) -> float:
        total = len(y)
        left_y = [y[i] for i in left_indices]
        right_y = [y[i] for i in right_indices]

        weighted_impurity = (
            len(left_y) / total * self._gini(left_y) + len(right_y) / total * self._gini(right_y)
        )
        return parent_impurity - weighted_impurity

    def _gini(self, y: list[Any]) -> float:
        counts = Counter(y)
        total = len(y)
        return 1.0 - sum((count / total) ** 2 for count in counts.values())

    def _majority_class(self, y: list[Any]) -> Any:
        counts = Counter(y)
        return counts.most_common(1)[0][0]

    def _predict_row(self, row: list[float], node: Node) -> Any:
        current = node
        while not current.is_leaf:
            if row[current.feature_index] <= current.threshold:
                current = current.left
            else:
                current = current.right
        return current.prediction

    def _print_node(self, node: Node, indent: str = "") -> None:
        if node.is_leaf:
            print(
                f"{indent}Leaf(prediction={node.prediction}, "
                f"samples={node.samples}, gini={node.impurity:.3f})"
            )
            return

        print(
            f"{indent}if feature[{node.feature_index}] <= {node.threshold:.3f} "
            f"(samples={node.samples}, gini={node.impurity:.3f})"
        )
        self._print_node(node.left, indent + "  ")
        print(f"{indent}else")
        self._print_node(node.right, indent + "  ")
