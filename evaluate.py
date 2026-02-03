from sklearn.metrics import classification_report

print(classification_report(y_test, svm.predict(X_test)))
