from decision_tree import DecisionTreeClassifier


def run_tests() -> None:
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.2, 1.1],
        [1.3, 0.9],
    ]
    y = ["A", "A", "B", "B", "B", "B"]

    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)

    predictions = tree.predict([[0.1, 0.2], [1.1, 0.8], [0.2, 0.9]])
    assert predictions == ["A", "B", "A"], predictions
    assert tree.score(X, y) >= 5 / 6

    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
