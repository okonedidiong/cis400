import sqlite3
import sys
import os
from numpy import genfromtxt
import numpy as np
import pandas as pd
import math
import nltk


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
    #df=pd.read_csv('reddit_opiates.csv', sep=',',header=None)
    df = pd.read_csv('reddit_opiates.csv', sep=',', header=None)
    data = df.values
    comments = data[0:10000, [0]] #opiates
    #comments = data[0:50, [1]] #lifestyle
    comments = comments.tolist()
    #print(comments)
    comments = [comment for sublist in comments for comment in sublist]
    #print(comments)

    return comments

def get_doc_term_matrix(comments, dictionary):
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(comment) for comment in comments[0:20]]
    #print(doc_term_matrix)
    #print(np.var(doc_term_matrix))
    return doc_term_matrix

def train_model():
    comments = get_comments()
    comments_clean = [clean(comment).split() for comment in comments]

    # Lemmatize
    lmtzr = WordNetLemmatizer()

    for i in range(423,430):
        print(comments_clean[i])
        print(lmtzr.lemmatize(comments[i]))
        print('-------------------')
        print('--------------------')
    return

    #comments = [lmtzr.lemmatize(comment) for comment in comments]

    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(comments_clean)
    doc_term_matrix = get_doc_term_matrix(comments_clean,dictionary)
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel
    # Running and Training LDA model on the document term matrix.
    lda_model = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=50)
    # Save model
    lda_model.save('lda_model')
    print(lda_model.print_topics(num_topics=10, num_words=3))
    #'''

def find_top_n(cluster_number,top_n, comment_map):
    list = []
    next_threshold = 0
    threshold = 0
    list_count = 0
    min_prob = 0
    for comment in comment_map:
        if comment_map[comment][0] == cluster_number: #Extract only from the cluster specified
            if len(list) <= top_n:
                list.append(comment)
                if comment_map[comment][1] < min_prob:
                    min_prob = comment_map[comment][1]
                    threshold = min_prob
            else:
                prob = comment_map[comment][1]
                if prob > threshold:
                    for x in list: #remove smallest probability
                        comment_map[comment][1] == threshold
                        list.remove(x)
                    list.append(comment)
                    probs = [comment_map[x][1] for x in list]

                    threshold = min(probs)
    return list

def use_model():
    lda_model = gensim.models.ldamodel.LdaModel.load('lda_model')
    print(lda_model.print_topics(num_topics=10, num_words=10))
    num_clusters = 10
    top_n = 5
    comments = get_comments()
    comments_clean = [clean(comment).split() for comment in comments]
    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(comments_clean)

    #print(lda_model.print_topics(num_topics=3, num_words=5))

    #print(lda_model.print_topics(num_topics=5, num_words=5))

    map = {}

    for i in range(0, len(comments)):
        bag_of_words = []
        comment = comments[i]
        #print('comment: ', comment)

        for word in comment:
            bag_of_words.append(word)


        bag_of_words = comment.split()


        if len(bag_of_words) > 0:
            last_word = bag_of_words[len(bag_of_words)-1]
            if len(last_word) > 0 and last_word[len(last_word)-1] in [',',';','?','.']: #remove end punctuation
                last_word = last_word[:-1]
                bag_of_words[len(bag_of_words)-1] = last_word

        bow = dictionary.doc2bow(bag_of_words)
        possible_homes = lda_model.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)

        #print(possible_homes)

        home = -1
        max_probability = 0

        for x in possible_homes:
            if x[1] > max_probability:
                max_probability = x[1]
                home = x[0]

        #print(home)
        map[comment] = [home, max_probability]

    clusters = [[] for i in range(0,num_clusters)]

    #print(clusters)

    for item in map:
        clusters[map[item][0]].append(item)

    '''
    for i in range(0,num_clusters):
        print(clusters[i])
        print('____________')
    '''

    '''
    for x in clusters[2]:
        print(x)
    '''

    '''
    count = 0

    print(lda_model.print_topics(num_topics=10, num_words=3))
    print('---------------')

    test_cluster4 = find_top_n(cluster_number=4, top_n=top_n, comment_map = map)
    test_cluster5 = find_top_n(cluster_number=5, top_n=top_n, comment_map=map)
    test_cluster6 = find_top_n(cluster_number=6, top_n=top_n, comment_map=map)
    test_cluster7 = find_top_n(cluster_number=7, top_n=top_n, comment_map=map)

    for i in test_cluster4:
        count = count + 1
        print(str(count) + ': ' +i)
    print('-----------')
    for i in test_cluster5:
        count = count + 1
        print(str(count) + ': ' +i)
    print('-----------')
    for i in test_cluster6:
        count = count + 1
        print(str(count) + ': ' + i)
    print('-----------')
    for i in test_cluster7:
        count = count + 1
        print(str(count) + ': ' + i)
    '''


def main():
    print('Hello World!')

if __name__ == "__main__":
    train_model()
    #use_model()