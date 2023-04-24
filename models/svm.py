import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

from liar_preprocess import load_liar_clean_csv
from isot_preprocess import load_isot_clean_csv

# df = load_liar_clean_csv()
df = load_isot_clean_csv(11000)

X = df['statement'].values.astype('U')

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

# Vectorize training data
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=7000)
X_train_vectors = vectorizer.fit_transform(X_train)

# Train SVM model
svm = SVC(kernel='rbf', gamma='auto', C=1.0)

svm.fit(X_train_vectors, y_train)

# Vectorize test data
X_train_vectors = vectorizer.transform(X_test)

# Test model
y_pred = svm.predict(X_train_vectors)

# Evaluate model
print('SVM Model Performance Evaluation')
print('-'*30)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted')}\n")
print('Classification Report:\n', classification_report(y_test, y_pred, zero_division=0))
