import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle

from liar_preprocess import load_liar_clean_csv
from isot_preprocess import load_isot_clean_csv

# df = load_liar_clean_csv()
df = shuffle(load_isot_clean_csv())

df = df.dropna()

# Use LabelEncoder to encode the labels
le = LabelEncoder()
le.fit(df['label'])
df['encoded_label'] = le.transform(df['label'])
# df['encoded_label'] = df['encoded_label'].astype(str)
df['encoded_label'] = df['encoded_label'].apply(lambda x: str(x))
# df['encoded_label'] = df['encoded_label'].astype(str).replace('\.0', '', regex=True)

# df = df.dropna()

# na_mask = df.isna().any(axis=1)
# na_rows = df[na_mask]

# print(na_rows)

# df['encoded_label'] = df['encoded_label'].apply(lambda x: str(x) if type(x) != str else x)
# print(df[df['encoded_label'].apply(lambda x: type(x) != str)])

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=56)

train_data['encoded_label'] = train_data['encoded_label'].astype(str)
test_data['encoded_label'] = test_data['encoded_label'].astype(str)

# Define the features and label sequences for the CRF model
def word2features(sent, i):
    word = sent[i][0]
    pos_tag = nltk.pos_tag([word])[0][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'pos_tag': pos_tag,
    }
    if i > 0:
        prev_word = sent[i-1][0]
        prev_pos_tag = nltk.pos_tag([prev_word])[0][1]
        features.update({
            'prev_word.lower()': prev_word.lower(),
            'prev_word.istitle()': prev_word.istitle(),
            'prev_word.isupper()': prev_word.isupper(),
            'prev_pos_tag': prev_pos_tag,
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        next_word = sent[i+1][0]
        next_pos_tag = nltk.pos_tag([next_word])[0][1]
        features.update({
            'next_word.lower()': next_word.lower(),
            'next_word.istitle()': next_word.istitle(),
            'next_word.isupper()': next_word.isupper(),
            'next_pos_tag': next_pos_tag,
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

# Convert the training and testing sets to sequences of features and labels
train_sents = [list(zip(train_data['statement'].iloc[i].split(), train_data['encoded_label'].astype(str).iloc[i].split())) for i in range(len(train_data))]
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

test_sents = [list(zip(test_data['statement'].iloc[i].split(), test_data['encoded_label'].astype(str).iloc[i].split())) for i in range(len(test_data))]
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]


# Train the CRF model
crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=1000, all_possible_transitions=True)
crf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = crf.predict(X_test)

# Evaluate the performance of the model
labels = list(crf.classes_)
if 'O' in labels:
    labels.remove('O')

print(metrics.flat_classification_report(y_test, y_pred))
print('-'*30)
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))