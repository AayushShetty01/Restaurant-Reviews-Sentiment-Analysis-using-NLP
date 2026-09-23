# Restaurant Review Sentiment Analysis

A reproducible Python/NLP pipeline for classifying restaurant reviews as positive or negative.

## Pipeline
1. Load a labelled CSV dataset.
2. Clean review text with regular expressions, stop-word removal, and stemming.
3. Split the data into training and test sets.
4. Fit TF-IDF features on the training data only.
5. Train Logistic Regression and Linear SVM classifiers.
6. Report accuracy and weighted F1-score on the held-out test set.

## Expected dataset format
The training CSV must contain two columns:

```text
review,sentiment
"The food was excellent",positive
"The service was slow",negative
```

The dataset is intentionally not committed unless its licence permits redistribution.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows activate with `.venv\\Scripts\\activate`.

## Run
From the repository root:
```bash
python -m src.models --data data/restaurant_reviews.csv
```

Use `--help` to see available options.

## Design notes
TF-IDF is fitted only on the training split to prevent information leakage. Evaluation reports both accuracy and F1-score.

## Project status
Educational NLP project demonstrating preprocessing, feature extraction, supervised classification, and evaluation.
