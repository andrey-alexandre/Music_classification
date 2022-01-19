import re
import string
from datetime import datetime
import pandas as pd
import numpy as np
import spacy
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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

X, y = shuffle(X, y)
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
kf = KFold(n_splits=10)
kf.get_n_splits(X)

names = [
    "Logistic Regression",
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    # "Neural Net",
    # "AdaBoost"
]

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # MLPClassifier(alpha=1, max_iter=1000),
    # AdaBoostClassifier()
]

clf_dict = dict()
time_dict = dict()
kfold_predicted_dict = {name: list() for name in names}
kfold_score_dict = {name: list() for name in names}
kfold_acc_dict = {name: list() for name in names}
kfold_recall_dict = {name: list() for name in names}
kfold_cm_dict = {name: list() for name in names}
for name, clf in zip(names, classifiers):
    i = 0
    time_dict[name] = list()
    time_dict[name].append(datetime.now())
    pipe = Pipeline([("cleaner", predictors()),
                     ('vectorizer', tfidf_vector),
                     ('classifier', clf)])

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pipe.fit(X_train, y_train)
        clf_dict[name] = pipe

        # Predicting with a test dataset
        kfold_score_dict[name].append(pipe.score(X_test, y_test))
        kfold_predicted_dict[name].append(pipe.predict(X_test))

        # Model Accuracy
        kfold_acc_dict[name].append(metrics.accuracy_score(y_test, kfold_predicted_dict[name][i]))
        kfold_cm_dict[name].append(metrics.confusion_matrix(y_test, kfold_predicted_dict[name][i]))
        kfold_recall_dict[name].append(metrics.recall_score(y_test, kfold_predicted_dict[name][i], average='macro'))

        i += 1
    time_dict[name].append(datetime.now())
print("Terminou de rodar!")

score_dict = {name: list() for name in names}
acc_dict = {name: list() for name in names}
recall_dict = {name: list() for name in names}
for name in names:
    score_dict[name] = np.mean(kfold_score_dict[name])
    acc_dict[name] = np.mean(kfold_acc_dict[name])
    recall_dict[name] = np.mean(kfold_recall_dict[name])

for name in names:
    print(f"O tempo de execução para o modelo {name} é ", time_dict[name][1] - time_dict[name][0])
    start = datetime.now()
    clf_dict[name].predict(X_test)
    end = datetime.now()
    print(f"O tempo de estimação para o modelo {name} é ", end - start)
