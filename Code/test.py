import spacy
import pandas as pd

nlp = spacy.load("pt_core_news_lg")


# New stop words list
customize_stop_words = ['\n', ' \n', ' \n']

# Mark them as stop words
for w in customize_stop_words:
    nlp.vocab[w].is_stop = True

text = pd.read_csv('./Data/gospel.csv')['lyric'][2]
doc = nlp(text)
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
list(doc.sents)


for ent in doc.ents:
    print(ent.text, ent.label_)

[token.lemma_.lower().strip() for token in nlp(text) if not token.is_stop and not token.is_punct and not token.is_space]