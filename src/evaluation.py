from sklearn.metrics import accuracy_score, classification_report, f1_score


def evaluate_model(y_true, y_pred):
    """Return standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "report": classification_report(y_true, y_pred),
    }
