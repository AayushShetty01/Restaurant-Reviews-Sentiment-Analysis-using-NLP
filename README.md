# Sentiment Analysis of Restaurant Reviews using NLP

## Project Overview
This project implements an NLP-based sentiment analytics system to classify
restaurant reviews as positive or negative. The system processes unstructured
text data using linguistic preprocessing and applies supervised machine learning
models to extract actionable insights from customer feedback.

The project demonstrates end-to-end NLP pipeline design, feature engineering,
model training, and empirical evaluation.

---

## Objectives
- Build a scalable text preprocessing pipeline for unstructured data
- Apply supervised learning techniques for sentiment classification
- Compare linear classifiers in high-dimensional text feature space
- Evaluate model performance using standard machine learning metrics

---

## Dataset
- Source: Public restaurant review dataset (e.g., Yelp / Kaggle)
- Size: 10,000+ text reviews
- Labels: Positive / Negative sentiment

---

## Methodology

### 1. Text Preprocessing
- Tokenization
- Stop-word removal
- Stemming (Porter Stemmer)
- Noise filtering using regular expressions

Implemented using **NLTK** to reduce vocabulary size and improve model generalization.

---

### 2. Feature Extraction
- TF-IDF vectorization
- Vocabulary capped at 5,000 features

This step transforms symbolic text data into numerical representations suitable
for statistical learning models.

---

### 3. Supervised Learning Models
- Logistic Regression (baseline linear classifier)
- Linear Support Vector Machine (margin-based classifier)

---

### 4. Evaluation Strategy
- 80/20 train-test split
- Accuracy and F1-score for model comparison

---

## Results

| Model | Accuracy |
|------|----------|
| Logistic Regression | 76% |
| Linear SVM | **78%** |

Linear SVM demonstrated improved performance due to better margin optimization
in sparse TF-IDF feature space.

---

## Technologies
- Python 3
- Pandas, NumPy
- NLTK
- Scikit-learn
- Git & GitHub

---

## How to Run
```bash
pip install -r requirements.txt
python src/models.py
