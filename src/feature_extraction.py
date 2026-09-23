from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features)


def extract_features(corpus, vectorizer=None):
    """Fit a TF-IDF vectorizer on corpus and return the transformed matrix."""
    vectorizer = vectorizer or build_vectorizer()
    features = vectorizer.fit_transform(corpus)
    return features, vectorizer
