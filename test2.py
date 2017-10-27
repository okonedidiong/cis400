import sqlite3
import sys
import os
from numpy import genfromtxt
import numpy as np
import pandas as pd
import math

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora


def clean(doc):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def docs_example():
    doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
    doc2 = "My father spends a lot of time driving my sister around to dance practice."
    doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
    doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
    doc5 = "Health experts say that Sugar is not good for your lifestyle."
    docs_complete = [doc1, doc2, doc3, doc4, doc5]
    return docs_complete

def get_comments():
    df=pd.read_csv('lifestyle_food.csv', sep=',',header=None)
    data = df.values
    comments = data[:, [1]]
    comments = comments.tolist()
    comments = [comment for sublist in comments for comment in sublist]
    for i in range(len(comments)):
        if type(comments[i]) == float:
            comments[i] = str(0)
    return comments

def get_doc_term_matrix(comments, dictionary):
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(comment) for comment in comments[0:20]]
    print(doc_term_matrix)
    print(np.var(doc_term_matrix))
    return doc_term_matrix

def train_model():
    comments = get_comments()
    comments_clean = [clean(comment).split() for comment in comments]
    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(comments_clean)
    doc_term_matrix = get_doc_term_matrix(comments_clean,dictionary)
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel
    # Running and Training LDA model on the document term matrix.
    lda_model = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
    # Save model
    lda_model.save('lda_model')
    print(lda_model.print_topics(num_topics=3, num_words=3))
    #'''

def use_model():
    lda_model = gensim.models.ldamodel.LdaModel.load('lda_model')
    #print(lda_model.print_topics(num_topics=3, num_words=5))
    print(lda_model.print_topics(num_topics=3, num_words=3))
    print(lda_model.print_topic(1, topn=15))



def main():
    print('Hello World!')

if __name__ == "__main__":
    use_model()