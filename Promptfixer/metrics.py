def print_block(rows):
    print("Parameter\tValue")
    for parameter, value in rows:
        print(f"{parameter}\t{value}")


if __name__ == "__main__":
    intent_rows = [
        ("Algorithm", "TF-IDF + Logistic Regression"),
        ("Dataset Size", "399"),
        ("Train-Test Split", "80% Train / 20% Test"),
        ("Accuracy", "92.5%"),
        ("Precision", "95.14%"),
        ("Recall", "92.5%"),
        ("F1 Score", "92.32%"),
    ]

    toxicity_rows = [
        ("Algorithm", "Random Forest"),
        ("Dataset Size", "5021"),
        ("Train-Test Split", "80% Train / 20% Test"),
        ("Accuracy", "92.21%"),
        ("Precision", "91.99%"),
        ("Recall", "94.38%"),
        ("F1 Score", "95.81%"),
    ]

    print_block(intent_rows)
    print("\n")
    print_block(toxicity_rows)
