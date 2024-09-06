from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)

word_index = imdb.get_word_index()
index_to_word = {index + 3: word for word, index in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"
index_to_word[3] = "<UNUSED>"

def decode_review(encoded_review):
    return ' '.join([index_to_word.get(i, '?') for i in encoded_review])

decoded_X_train = [decode_review(review) for review in X_train]
decoded_X_test = [decode_review(review) for review in X_test]

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

processed_X_train = [preprocess_text(text) for text in decoded_X_train]
processed_X_test = [preprocess_text(text) for text in decoded_X_test]

vectorizer = CountVectorizer(max_features=10000)
X_train_bow = vectorizer.fit_transform(processed_X_train)
X_test_bow = vectorizer.transform(processed_X_test)

model = MultinomialNB()
model.fit(X_train_bow, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test_bow)

# Оценка модели
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
