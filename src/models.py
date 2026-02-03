import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from preprocessing import clean_text
from feature_extraction import extract_features

df = pd.read_csv("data/restaurant_reviews.csv")

df['cleaned_review'] = df['review'].apply(clean_text)

X, vectorizer = extract_features(df['cleaned_review'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=1000)
svm = SVC(kernel='linear')

lr.fit(X_train, y_train)
svm.fit(X_train, y_train)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr.predict(X_test)))
print("SVM Accuracy:", accuracy_score(y_test, svm.predict(X_test)))
