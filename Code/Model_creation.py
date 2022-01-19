import re
import string
import pandas as pd
import spacy
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump

df = pd.read_csv('./Data/full.csv', encoding='utf-8-sig')

# Create our list of punctuation marks
punctuations = string.punctuation
rep_punctuations = '!{2,}|"{2,}|#{2,}|\${2,}|%{2,}|&{2,}|\'{2,}|({2,}|){2,}|\*{2,}|+{2,}|,{2,}|-{2,}|\.{2,}|/{2,}|:{2,' \
                   '}|;{2,}|<{2,}|={2,}|>{2,}|?{2,}|@{2,}|[{2,}|\\{2,}|]{2,}|\^{2,}|_{2,}|`{2,}|{{2,}||{2,}|}{2,}|~ '

# Create our list of white spaces
whitespaces = string.whitespace

# Create our list of stopwords
nlp = spacy.load('pt_core_news_sm')
stop_words = spacy.lang.pt.stop_words.STOP_WORDS


# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.lower().strip() for word in mytokens]

    # Removing stop words
    mytokens = [word for word in mytokens if
                word not in stop_words and word not in punctuations and word not in whitespaces]

    # Removing strange tokens
    mytokens = [word for word in mytokens if
                not re.match('\.{2,}', word) and not re.match('^\(*[0-9]x', word) and not re.match('-+.',
                                                                                                   word) and not re.match(
                    '\(.', word)]

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


class feature_extraction(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))
tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)

X = df['lyric']  # the features we want to analyze
y = df['Class']  # the labels, or answers, we want to test against

pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', tfidf_vector),
                 ('classifier', LogisticRegression())])
pipe.fit(X, y)
dump(pipe, './Docs/Models/lr_clf.joblib')
