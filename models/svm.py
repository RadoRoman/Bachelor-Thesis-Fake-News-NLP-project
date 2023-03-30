import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import sys
# sys.path.append('C:\\Work\\Bachelor-Thesis-Fake-News-NLP-project\\utils.py')
from utils import tsv_reader, path_to_liar_dir

# data_frame_LIAR = tsv_reader(path_to_liar_dir() + 'merged.tsv')
data_frame_LIAR = pd.read_csv(path_to_liar_dir() + 'merged.tsv', sep='\t', usecols=[1, 2])

data_frame_LIAR.columns = ['label', 'statement']

print(data_frame_LIAR.head())

# Convert the labels into numerical values (1 for true and 0 for false, pants on fire, etc.)
data_frame_LIAR['label'] = np.where(data_frame_LIAR['label'].str.contains('true'), 1, 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_frame_LIAR['statement'], data_frame_LIAR['label'], test_size=0.2, random_state=42)

# Convert the text data into numerical features using HashingVectorizer and TfidfTransformer
count_vect = HashingVectorizer(stop_words='english')
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Train the SVM model
clf = LinearSVC()
clf.fit(X_train_tfidf, y_train)

# Test the model and generate a classification report
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
print ("-"*30)
# Print accuracy score and confusion matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print ("-"*30)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))