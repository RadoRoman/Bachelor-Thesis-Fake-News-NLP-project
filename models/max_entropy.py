import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

df = pd.read_csv('liar_dataset/train.tsv', sep='\t', header=None, names=['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire', 'venue'])

# Remove unnecessary columns
df = df[['statement', 'label']]

# Clean the text
def clean_text(text):
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'#\w+', '', text) # Remove hashtags
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuations and special characters
    text = text.lower() # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words]) # Remove stop words
    text = ' '.join([stemmer.stem(word) for word in text.split()]) # Stem the words
    return text

def pos_tagging(text):
    pos_tags = pos_tag(word_tokenize(text))
    return ' '.join([word + '_' + tag for word, tag in pos_tags])

df['statement'] = df['statement'].apply(clean_text)

# Bag of words feature
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(df['statement'])

# N-grams feature
vectorizer_ngrams = CountVectorizer(ngram_range=(1,3))
X_ngrams = vectorizer_ngrams.fit_transform(df['statement'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_ngrams, df['label'], test_size=0.2, random_state=42)

# Train the Maximum Entropy model
max_ent_model = LogisticRegression(max_iter=10000)
max_ent_model.fit(X_train, y_train)

from sklearn.metrics import classification_report

# Predict the labels of the testing set
y_pred = max_ent_model.predict(X_test)

# Calculate the metrics
print(classification_report(y_test, y_pred))