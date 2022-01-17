import re
import string
import pandas as pd
import spacy


df = pd.read_csv('./Data/full.csv', encoding='utf-8-sig')

# Create our list of punctuation marks
punctuations = string.punctuation
rep_punctuations = '!{2,}|"{2,}|#{2,}|\${2,}|%{2,}|&{2,}|\'{2,}|({2,}|){2,}|\*{2,}|+{2,}|,{2,}|-{2,}|\.{2,}|/{2,}|:{2,' \
                   '}|;{2,}|<{2,}|={2,}|>{2,}|?{2,}|@{2,}|[{2,}|\\{2,}|]{2,}|\^{2,}|_{2,}|`{2,}|{{2,}||{2,}|}{2,}|~ '

# Create our list of white spaces
whitespaces = string.whitespace

# Create our list of stopwords
nlp = spacy.load('pt_core_news_lg')
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

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()


nlp = spacy.load('pt_core_news_lg')
# print(len(b))
# print(len(list(b.sents)))
# print([sent for sent in b.sents])

df['lyric_doc'] = df['lyric'].apply(lambda x: nlp(x))
df['word_count'] = df['lyric_doc'].apply(lambda x: len(x))
df['sent_count'] = df['lyric_doc'].apply(lambda x: len(list(x.sents)))

df[['word_count', 'Class']].boxplot()

df.to_csv('./Data/plot.csv')