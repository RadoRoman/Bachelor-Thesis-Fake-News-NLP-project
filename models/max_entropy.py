import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


from liar_preprocess import load_liar_clean_csv
from isot_preprocess import load_isot_clean_csv


df = load_liar_clean_csv()
# df = load_isot_clean_csv(7000)

df['statement'] = df['statement'].values.astype('U')

# # Bag of words feature
# vectorizer_bow = CountVectorizer()
# X_bow = vectorizer_bow.fit_transform(df['statement'])

# N-grams feature
vectorizer_ngrams = CountVectorizer(ngram_range=(1,3))
X_ngrams = vectorizer_ngrams.fit_transform(df['statement'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_ngrams, df['label'], test_size=0.2, random_state=19)

# Create Maximum Entropy model
me_model = LogisticRegression(max_iter=10000)

# Define hyperparameters for tuning
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2']
}

# Perform Grid Search to find the optimal hyperparameters
grid_search = GridSearchCV(estimator=me_model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best hyperparameters: ", grid_search.best_params_)
me_model_best = grid_search.best_estimator_
me_model_best.fit(X_train, y_train)

y_pred = me_model_best.predict(X_test)
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))