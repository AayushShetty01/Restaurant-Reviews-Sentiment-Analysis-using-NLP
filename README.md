# Restaurant Review Sentiment Analysis

A reproducible Python/NLP pipeline for classifying restaurant reviews as positive or negative using classical machine-learning techniques.

## Why this project

The project demonstrates an end-to-end text-classification workflow rather than a single training script:

Raw reviews -> Text cleaning -> Train/test split -> TF-IDF -> Logistic Regression / Linear SVM -> Accuracy + F1

## Technical approach

1. Load a labelled CSV dataset.
2. Validate the expected review and sentiment columns.
3. Normalize review text with regular expressions, stop-word removal, and stemming.
4. Split data into training and test sets using a fixed random seed and stratification.
5. Fit TF-IDF features on the training data only.
6. Train Logistic Regression and Linear SVM classifiers.
7. Evaluate predictions using accuracy and weighted F1-score.

Using a scikit-learn Pipeline keeps feature fitting inside the training workflow and avoids test-set information leaking into model training.

## Technology

- Language: Python
- Data: pandas, NumPy
- NLP: NLTK
- Machine learning: scikit-learn
- Testing: pytest
- CI: GitHub Actions

## Repository structure

    .
    ├── src/
    │   ├── preprocessing.py
    │   ├── feature_extraction.py
    │   ├── evaluation.py
    │   └── models.py
    ├── tests/
    │   └── test_pipeline.py
    ├── requirements.txt
    └── .github/
        └── workflows/
            └── ci.yml

## Dataset format

The training CSV should contain:

    review,sentiment
    "The food was excellent",positive
    "The service was slow",negative

The dataset itself is not committed unless redistribution is permitted by its licence.

## Installation

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

On Windows activate with .venv/Scripts/activate.

## Run

    python -m src.models --data data/restaurant_reviews.csv

Use --help to view available command-line options.

## Testing

Run:

    python -m pytest -q

GitHub Actions runs the test suite on Python 3.11 and 3.12 for pushes and pull requests.

## Reproducibility

The train/test split uses random_state=42 and stratification. Model metrics are calculated against a held-out test set.

The repository intentionally avoids claiming benchmark performance until the dataset and experiment configuration are explicitly recorded.

## Project status

Educational NLP / machine-learning project. The current implementation focuses on a clean, testable classical NLP pipeline. Future extensions could include hyperparameter tuning, cross-validation, confusion-matrix reporting, experiment tracking, and a small inference API.
