import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk.corpus import stopwords


stop_words = []
stop_words.extend(stopwords.words([
    'hungarian','swedish','norwegian','finnish',
    'portuguese','spanish', 'danish','romanian',
    'dutch', 'german', 'english', 'russian', 'french', 'italian']))


stop_words.extend(
    ['yo','dont','don\'t','uh', 'got', 'oh','ooh','us', 'im', 'na', 'from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be',
        'know', 'good', 'go', 'get', 'ah', 'bout','yeah','le','ayy','u','eh','wa',
        'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot',
        'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come',
        'really', 'la', 'la la', 'ich', 'hey', 'hey hey', 'nie', 'mi','never' 'ya', 'yuh','uuu','one', 'give','cos', 'back',
        'nao', 'shit', 'bitches', 'one', 'two','three', 'four', 'five', 'll','whoa', 'nae','ass','ga', 'printr', "let",
        'away','here','when','how','who','is','not','doo','fur','mehr', 'ooo', 'mal', 'schon', 'gimme','tak', 'ohh', 'ey','oo', 'hi', 'lil','yang',
        'chto', 'kak', 'ty', 'za', 'bo', 'gon', 'tryna', 'gotta', 'gonna', 'never', 'wanna','mo', 'ang', 'ay', 'jak', 'ko', 'po', 'aku', 'co', 'kau', 'ka',
        'ku', 'ng','eto','first','second','third','fourth','on', 'ona', 'ono', 'vsak', 'bu', 'mne', 'mnie', 'jest','bi', 'vse','ahh',  'ten', 'mam', 'shi',
        'sam','like', 'dey', 'nan', 'ye', 'tebia', 'juz', 'tam','yi','tylko', 'chce', 'dick', 'bez', 'nam', 'kto','menia', 'net', 'tut', 'od','lass','ab','dla','nein', 'oooh', 'hoo', 'ho','ooooh', 'aah', 'ah', 'yah', 'ib', 'gib', 'wann', 'ik', 'bir', 'gibi', 'beni',
        'bana', 'seni', 'yok', 'ama', 'benim', 'cok', 'jah', 'uer', 'pom', 'neol', 'deo', 'nal', 'nuk', 'neoreul', 'naega', 'neo', 'geu', 'woo', 'nareul', 'ke',
        'pa', 'bam', 'zai', 'xiang','hella', 'dhe', 'diyo', 'bizi', 'gep', 'hep','dera']  
        )

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower() # Convert to lowercase
    re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    text = re.sub(r'#\w+', '', text) # Remove hashtags
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuations and special characters
    text = ' '.join([word for word in text.split() if word not in stop_words]) # Remove stop words
    text = ' '.join([stemmer.stem(word) for word in text.split()]) # Stem the words
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def main():
    pass

if __name__ == "__main__":
    main()