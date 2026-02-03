from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from preprocess import preprocess_data

df = preprocess_data("data/restaurant_reviews.csv")

X = df['cleaned_review']
y = df['sentiment']

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

lr = LogisticRegression()
lr.fit(X_train, y_train)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr.predict(X_test)))
print("SVM Accuracy:", accuracy_score(y_test, svm.predict(X_test)))
