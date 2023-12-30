from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import textwrap
import nltk
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# nltk.download('punkt')
# nltk.download('stopwords')

df = pd.read_excel('News_snippet.xlsx')


def summarize(text, factor=0.15):
    # extract sentences
    sents = nltk.sent_tokenize(text)

    # perform tf-idf
    featurizer = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        norm='l1')
    X = featurizer.fit_transform(sents)

    # compute similarity matrix
    G = cosine_similarity(X)

    # normalize similarity matrix
    G /= G.sum(axis=1, keepdims=True)

    # uniform transition matrix
    U = np.ones_like(G) / len(G)

    # smoothed similarity matrix
    A = (1 - factor) * G + factor * U

    # find the limiting / stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(A.T)

    # compute scores
    scores = eigenvecs[:, 0] / eigenvecs[:, 0].sum()

    # sort the scores
    sort_idx = np.argsort(-scores)

    # print summary
    for i in sort_idx[:5]:
        print(wrap("%.2f: %s" % (scores[i], sents[i])))

def wrap(x):
  return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)

summarize(df['Text'][0])