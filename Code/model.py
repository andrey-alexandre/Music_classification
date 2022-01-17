import re
import string
from datetime import datetime
import numpy as np
import pandas as pd
import spacy
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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

# Custom transformer using spaCy
class feature_extraction(TransformerMixin):
    def transform(self, X, **transform_params):
        df = pd.DataFrame.sparse.from_spmatrix(X)

        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))
tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)

X = df['lyric']  # the features we want to analyze
y = df['Class']  # the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# tfidf_vector.fit_transform(X_train)
# print(sorted(tfidf_vector.get_feature_names()))
#
# lr = LogisticRegression()
#
# # Create pipeline using Bag of Words
# lr_pipe = Pipeline([("cleaner", predictors()),
#                  ('vectorizer', bow_vector),
#                  ('classifier', lr)])
#
# # model generation
# lr_pipe.fit(X_train, y_train)
#
# # Predicting with a test dataset
# lr_predicted = lr_pipe.predict(X_test)
#
# # Model Accuracy
# print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, lr_predicted))
# # print("Logistic Regression Precision:", metrics.precision_score(y_test, lr_predicted))
# print("Logistic Regression Recall:", metrics.recall_score(y_test, lr_predicted, average='micro'))
#
#
# svm = LinearSVC()
#
# # Create pipeline using Bag of Words
# svm_pipe = Pipeline([("cleaner", predictors()),
#                  ('vectorizer', bow_vector),
#                  ('classifier', svm)])
#
# # model generation
# svm_pipe.fit(X_train, y_train)
#
# # Predicting with a test dataset
# svm_predicted = svm_pipe.predict(X_test)
#
# # Model Accuracy
# print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, svm_predicted))
# # print("Logistic Regression Precision:", metrics.precision_score(y_test, lr_predicted))
# print("Logistic Regression Recall:", metrics.recall_score(y_test, svm_predicted, average='micro'))
#
# nb = MultinomialNB()
#
# # Create pipeline using Bag of Words
# nb_pipe = Pipeline([("cleaner", predictors()),
#                  ('vectorizer', bow_vector),
#                  ('classifier', nb)])
#
# # model generation
# nb_pipe.fit(X_train, y_train)
#
# # Predicting with a test dataset
# nb_predicted = nb_pipe.predict(X_test)
#
# # Model Accuracy
# print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, nb_predicted))
# # print("Logistic Regression Precision:", metrics.precision_score(y_test, lr_predicted))
# print("Logistic Regression Recall:", metrics.recall_score(y_test, nb_predicted, average='micro'))

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

names = [
    "Logistic Regression",
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    # "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    # "Naive Bayes",
    # "QDA",
]

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
]

clf_dict = dict()
time_dict = dict()
predicted_dict = dict()
score_dict = dict()
acc_dict = dict()
recall_dict = dict()
cm_dict = dict()
for name, clf in zip(names, classifiers):
    time_dict[name] = list()
    time_dict[name].append(datetime.now())
    pipe = Pipeline([("cleaner", predictors()),
                     ('vectorizer', tfidf_vector),
                     ('classifier', clf)])

    pipe.fit(X_train, y_train)
    clf_dict[name] = pipe
    # Predicting with a test dataset
    score_dict[name] = pipe.score(X_test, y_test)
    predicted_dict[name] = pipe.predict(X_test)

    # Model Accuracy
    acc_dict[name] = metrics.accuracy_score(y_test, predicted_dict[name])
    cm_dict[name] = metrics.confusion_matrix(y_test, predicted_dict[name])
    recall_dict[name] = metrics.recall_score(y_test, predicted_dict[name], average=None)
    time_dict[name].append(datetime.now())

for name in names:
    print(f"O tempo de execução para o modelo {name} é ", time_dict[name][1] - time_dict[name][0])
    start = datetime.now()
    clf_dict[name].predict(X_test)
    end = datetime.now()
    print(f"O tempo de estimação para o modelo {name} é ", end - start)
