import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import clean_text


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"review", "sentiment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["cleaned_review"] = df["review"].map(clean_text)
    return df


def evaluate(name, model, x_test, y_test):
    predictions = model.predict(x_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "f1_weighted": f1_score(y_test, predictions, average="weighted"),
    }


def run(data_path: Path):
    df = load_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        df["cleaned_review"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }

    results = []
    for name, classifier in models.items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)
        results.append(evaluate(name, pipeline, x_test, y_test))

    for result in results:
        print(
            f"{result['model']}: "
            f"accuracy={result['accuracy']:.4f}, "
            f"f1_weighted={result['f1_weighted']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Train restaurant review sentiment models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/restaurant_reviews.csv"),
        help="Path to a CSV containing review and sentiment columns.",
    )
    args = parser.parse_args()
    run(args.data)


if __name__ == "__main__":
    main()
