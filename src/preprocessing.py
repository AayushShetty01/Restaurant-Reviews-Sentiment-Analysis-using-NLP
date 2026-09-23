import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_STOP_WORDS = None
_STEMMER = PorterStemmer()


def _get_stop_words():
    global _STOP_WORDS
    if _STOP_WORDS is None:
        try:
            _STOP_WORDS = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _STOP_WORDS = set(stopwords.words("english"))
    return _STOP_WORDS


def clean_text(text: str) -> str:
    """Normalize a review for classical NLP models."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    words = text.split()
    stop_words = _get_stop_words()
    return " ".join(_STEMMER.stem(word) for word in words if word not in stop_words)
