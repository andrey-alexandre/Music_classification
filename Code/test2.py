import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re
import string
import spacy

df = pd.read_csv('./Data/full.csv', encoding='utf-8-sig')

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of white spaces
whitespaces = string.whitespace

# Create our list of stopwords
nlp = spacy.load('pt_core_news_sm')
stop_words = spacy.lang.pt.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = spacy.lang.pt.Portuguese()


# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.lower().strip() for word in mytokens]

    # Removing stop words
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations and word not in whitespaces]

    # Removing strange tokens
    mytokens = [word for word in mytokens if not re.match('\.{2,}', word) and not re.match('\([0-9]x', word)]

    # return preprocessed list of tokens
    return mytokens

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()


bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))
tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)


X = df['lyric']  # the features we want to analyze
y = df['Class']  # the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

tfidf_vector.fit_transform(X_train)
print(sorted(tfidf_vector.get_feature_names()))

classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train, y_train)

from sklearn import metrics

# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:", metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:", metrics.recall_score(y_test, predicted))
