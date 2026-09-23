import pandas as pd
import pytest

from src.models import load_data
from src.preprocessing import clean_text


def test_clean_text_removes_stopwords_and_normalizes():
    result = clean_text("The food was AMAZING!!!")
    assert "the" not in result.split()
    assert "food" in result
    assert "amaz" in result


def test_load_data_requires_expected_columns(tmp_path):
    path = tmp_path / "reviews.csv"
    pd.DataFrame({"text": ["great"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_data(path)
