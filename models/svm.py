import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import random

from liar_preprocess import load_liar_clean_csv
from isot_preprocess import load_isot_clean_csv

RANDOM_STATE = random.randint(1,999)

choice = input('Liar [1] or Isot [2]? : ')

if choice == '1':
    df = load_liar_clean_csv() #LIAR Data-Set
    max_df = 0.25
    ngram_range = (1, 2)
    C = 1
    kernel = 'rbf'

if choice == '2':
    size = int(input('size of dataset? :'))
    df = load_isot_clean_csv(size) #ISOT Data-Set
    max_df = 0.5
    ngram_range = (1, 2)
    C = 10
    kernel = 'sigmoid'

df = df.dropna()

X = df['statement'].values.astype('U')

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

vectorizer = TfidfVectorizer(max_df=max_df, ngram_range=ngram_range, max_features=7000) #Data Vectorization
X_train_vectors = vectorizer.fit_transform(X_train)


svm = SVC(kernel=kernel, gamma='auto', C=C,cache_size = 500, random_state=RANDOM_STATE)
svm.fit(X_train_vectors, y_train)

X_train_vectors = vectorizer.transform(X_test)
y_pred = svm.predict(X_train_vectors)

print('SVM Model Performance Evaluation')
print('-'*30)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted')}\n")
print('Classification Report:\n', classification_report(y_test, y_pred, zero_division=0))
