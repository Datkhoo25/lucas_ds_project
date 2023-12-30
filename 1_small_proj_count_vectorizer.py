import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
import matplotlib.pyplot as plt

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

df = pd.read_csv('bbc_text_cls.csv')
print(df.head())

inputs = df['text']
labels = df['labels']

# PLotting a histogram
labels.hist(figsize=(10, 5))
plt.show()

# Let's count how many percentage of non-zero
class findzeropercent:
    def __init__(self, xtrain):
        self.fzp = self.zero_over_prodshape(xtrain)

    def zero_over_prodshape(self, xtrain):
        ze = (xtrain != 0).sum()
        percentages = ze * 100 / np.prod(xtrain.shape)
        return percentages

inputs_train, inputs_test, ytrain, ytest = train_test_split(inputs, labels, random_state=100)

vectorizer = CountVectorizer()

xtrain = vectorizer.fit_transform(inputs_train)  # xtrain is a sparse matrix
xtest = vectorizer.transform(inputs_test)
findit = findzeropercent(xtrain)
print(findit.fzp)

model = MultinomialNB()
model.fit(xtrain, ytrain)
print("Model Score without stopword/default:", model.score(xtrain, ytrain))
print("Model Score without stopword/default::", model.score(xtest, ytest))
findit = findzeropercent(xtrain)
print(findit.fzp)

# with stopwords
vectorizer = CountVectorizer(stop_words='english')
xtrain = vectorizer.fit_transform(inputs_train)  # xtrain is a sparse matrix
xtest = vectorizer.transform(inputs_test)
model = MultinomialNB()
model.fit(xtrain, ytrain)
print("Model Score with built-in 'english' stopword:", model.score(xtrain, ytrain))
print("Model Score  with built-in 'english' stopword:", model.score(xtest, ytest))
findit = findzeropercent(xtrain)
print(findit.fzp)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self,doc):  #Call directly invoke it like a function #https://medium.com/@johnidouglasmarangon/using-call-method-to-invoke-class-instance-as-a-function-f61396145567#:~:text=The%20__call__%20method,to%20it%20as%20callable%20object.
        tokens = word_tokenize(doc)
        words_and_tags = nltk.pos_tag(tokens)
        return [self.wnl.lemmatize(word, pos=get_wordnet_pos(tag))\
                for word,tag in words_and_tags]

#With lemmatization
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer()) #Instead of a object LemmaTokenizer, you could pass in a function as well.
xtrain = vectorizer.fit_transform(inputs_train)
xtest = vectorizer.transform(inputs_test)
model = MultinomialNB()
model.fit(xtrain, ytrain)
print("Model Score  with lemmatization:", model.score(xtrain, ytrain))
print("Model Score  with lemmatization:", model.score(xtest, ytest))
findit = findzeropercent(xtrain)
print(findit.fzp)

class StemTokenizer:
    def __init__(self):
        self.porterlucluc = PorterStemmer()
    def __call__(self,doc):  #Call directly invoke it like a function #https://medium.com/@johnidouglasmarangon/using-call-method-to-invoke-class-instance-as-a-function-f61396145567#:~:text=The%20__call__%20method,to%20it%20as%20callable%20object.
        tokens = word_tokenize(doc)
        return [self.porterlucluc.stem(t) for t in tokens]

#With Stemming
vectorizer = CountVectorizer(tokenizer=StemTokenizer()) #Instead of a object LemmaTokenizer, you could pass in a function as well.
xtrain = vectorizer.fit_transform(inputs_train)
xtest = vectorizer.transform(inputs_test)
model = MultinomialNB()
model.fit(xtrain, ytrain)
print("Model Score with stemming:", model.score(xtrain, ytrain))
print("Model Score with stemming:", model.score(xtest, ytest))
findit = findzeropercent(xtrain)
print(findit.fzp)
def simple_tokenizer(s):
    return s.split()


# string split tokenizer
vectorizer = CountVectorizer(
    tokenizer=simple_tokenizer)  # Instead of a object LemmaTokenizer, you could pass in a function as well.
xtrain = vectorizer.fit_transform(inputs_train)
xtest = vectorizer.transform(inputs_test)
model = MultinomialNB()
model.fit(xtrain, ytrain)
print("Model Score simple split tokenizer:", model.score(xtrain, ytrain))
print("Model Score simple split tokenizer:", model.score(xtest, ytest))
findit = findzeropercent(xtrain)
print(findit.fzp)