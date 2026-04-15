from decision_tree import DecisionTreeClassifier


def main() -> None:
    # Toy "play tennis" style dataset with numeric-encoded features:
    # [temperature, humidity, windy]
    X = [
        [85, 85, 0],
        [80, 90, 1],
        [83, 78, 0],
        [70, 96, 0],
        [68, 80, 0],
        [65, 70, 1],
        [64, 65, 1],
        [72, 95, 0],
        [69, 70, 0],
        [75, 80, 0],
    ]
    y = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes"]

    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
    tree.fit(X, y)

    print("Tree:")
    tree.print_tree()

    print("\nPredictions:")
    test_points = [
        [66, 72, 1],
        [84, 88, 0],
        [71, 91, 0],
    ]
    predictions = tree.predict(test_points)
    for point, prediction in zip(test_points, predictions):
        print(f"{point} -> {prediction}")

    print(f"\nTraining accuracy: {tree.score(X, y):.2%}")


if __name__ == "__main__":
    main()
