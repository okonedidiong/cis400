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
import pickle

from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import preprocessing

import itertools



def clean(doc):
    manual_stop = open('manual_stop_words', 'r')
    manual_stop_list = [line.strip() for line in manual_stop]
    additional_stop = open('additional_stop_words', 'r')
    additional_stop_list = [line.strip() for line in additional_stop]
    internet_slang = open('internet_slang', 'r')
    internet_slang_list = [line.strip() for line in internet_slang]
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    if type(doc) != str:
        doc = str(doc)
        print('changed')
        print(type(doc))

    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    stop_free2 = " ".join([i for i in stop_free.split() if i not in manual_stop_list])
    internet_slang_free = " ".join([i for i in stop_free2.split() if i not in internet_slang_list])
    punc_free = ''.join(ch for ch in internet_slang_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    stop_free = " ".join([i for i in normalized.split() if i not in stop])
    stop_free2 = " ".join([i for i in stop_free.split() if i not in manual_stop_list])
    short_words_free = " ".join([i for i in stop_free2.split() if len(i) > 2])
    internet_slang_free = " ".join([i for i in short_words_free.split() if i not in internet_slang_list])
    additional_stop_free = " ".join([i for i in internet_slang_free.split() if i not in additional_stop_list])

    return additional_stop_free

def docs_example():
    doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
    doc2 = "My father spends a lot of time driving my sister around to dance practice."
    doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
    doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
    doc5 = "Health experts say that Sugar is not good for your lifestyle."
    docs_complete = [doc1, doc2, doc3, doc4, doc5]
    return docs_complete

def get_num_comments(model):
    if model == 'lda':
        df = pd.read_csv('mydata.csv', sep=',', header=None)
    elif model == 'depression_lda':
        df = pd.read_csv('depression_data_all.csv', sep=',', header=None)
    elif model == 'womens_health_lda':
        df = pd.read_csv('wh.csv', sep=',', header=None)
    elif model == 'derm_cannabis_lda':
        return len(get_cannabis_comments())

    data = df.values
    return len(data)


def get_comments(num_comments, model):
    if model == 'lda':
        df = pd.read_csv('mydata.csv', sep=',', header=None)
    elif model == 'depression_lda':
        df = pd.read_csv('depression_data_all.csv', sep=',', header=None)
    elif model == 'womens_health_lda':
        df = pd.read_csv('wh.csv', sep=',', header=None)
    elif model == 'derm_cannabis_lda':
        return get_cannabis_comments()
    else:
        return

    data = df.values
    print(len(data))
    comments = data[:num_comments, [0]]
    comments = comments.tolist()
    comments = [comment for sublist in comments for comment in sublist]
    comments = [str(comment) for comment in comments]
    return comments

def load_model(model):
    if model == 'lda':
        lda_model = gensim.models.ldamodel.LdaModel.load('lda_model')
    elif model == 'depression_lda':
        lda_model = gensim.models.ldamodel.LdaModel.load('depression_lda_model')
    elif model == 'womens_health_lda':
        lda_model = gensim.models.ldamodel.LdaModel.load('womens_health_lda_model')
    elif model == 'derm_cannabis_lda':
        lda_model = gensim.models.ldamodel.LdaModel.load('derm_cannabis_lda_model')
    else:
        return None

    return lda_model

def intersect(str1, list):
    for item in list:
        if item in str1:
            return True
    return False

def get_cannabis_comments():
    df = pd.read_csv('mydata.csv', sep=',', header=None)

    top_500 = pd.read_csv('cannabis_500.csv')
    top_500 = top_500.values
    top_500 = [t[1] for t in top_500]

    data = df.values
    comments = data[:, [0]]
    comments = comments.tolist()
    comments = [comment for sublist in comments for comment in sublist]
    comments = [str(comment) for comment in comments]

    cannabis_comments = []
    for c in comments:
        if intersect(c, top_500):
            cannabis_comments.append(c)


    '''
    w2v_model = pickle.load(open('word2vec_model', 'rb'))
    print(w2v_model.similar_by_vector('marijuana', 10))
    '''

    return cannabis_comments


def train_model(k, num_comments, model):

    # Load Reddit comments
    comments = get_comments(num_comments, model)

    # Clean Reddit comments
    comments_clean = [clean(comment).split() for comment in comments]

    # Creating the term dictionary of our courpus, where every unique term is assigned an index
    dictionary = corpora.Dictionary(comments_clean)

    # Create document term matrix from the dictionary created above
    doc_term_matrix = [dictionary.doc2bow(comment) for comment in comments_clean]


    if model == 'lda':
        #Creating the object for LDA model using gensim library
        Lda = gensim.models.ldamodel.LdaModel

        # Running and training LDA model on the document term matrix.
        lda_model = Lda(doc_term_matrix, num_topics=k, id2word = dictionary, passes=10)

        # Save model
        lda_model.save('lda_model')

        # Print the top 25 words from each topic
        for i in lda_model.print_topics(num_topics=k, num_words=25):
            print(i)

    if model == 'derm_cannabis_lda':
        # Creating the object for LDA model using gensim library
        Lda = gensim.models.ldamodel.LdaModel

        # Running and training LDA model on the document term matrix.
        lda_model = Lda(doc_term_matrix, num_topics=k, id2word=dictionary, passes=10)

        # Save model
        lda_model.save('derm_cannabis_lda_model')

        # Print the top 25 words from each topic
        for i in lda_model.print_topics(num_topics=k, num_words=25):
            print(i)

    if model == 'depression_lda':
        # Creating the object for LDA model using gensim library
        Lda = gensim.models.ldamodel.LdaModel

        # Running and training LDA model on the document term matrix.
        lda_model = Lda(doc_term_matrix, num_topics=k, id2word=dictionary, passes=10)

        # Save model
        lda_model.save('depression_lda_model')

        # Print the top 25 words from each topic
        for i in lda_model.print_topics(num_topics=k, num_words=50):
            print(i)

    if model == 'womens_health_lda':
        # Creating the object for LDA model using gensim library
        Lda = gensim.models.ldamodel.LdaModel

        # Running and training LDA model on the document term matrix.
        lda_model = Lda(doc_term_matrix, num_topics=k, id2word=dictionary, passes=10)

        # Save model
        lda_model.save('womens_health_lda_model')

        # Print the top 25 words from each topic
        for i in lda_model.print_topics(num_topics=k, num_words=50):
            print(i)

    if model == 'cannabis_lda':
        # Creating the object for LDA model using gensim library
        Lda = gensim.models.ldamodel.LdaModel

        # Running and training LDA model on the document term matrix.
        lda_model = Lda(doc_term_matrix, num_topics=k, id2word=dictionary, passes=10)

        # Save model
        lda_model.save('cannabis_lda_model')

        # Print the top 25 words from each topic
        for i in lda_model.print_topics(num_topics=k, num_words=200):
            print(i)

    elif model == 'hdp':
        # Creating the object for HDP model using gensim library
        Hdp = gensim.models.HdpModel

        #Running and training HDP model on the document term matrix
        hdp_model = Hdp(corpus=doc_term_matrix, id2word=dictionary)

        # Save model
        hdp_model.save('hdp_model')

        # Print the top 25 words from each topic
        for i in hdp_model.print_topics(num_topics=k, num_words=25):
            print(i)

    elif model == 'lsi':
        # Creating the object for HDP model using gensim library
        Lsi = gensim.models.LsiModel

        lsi_model = Lsi(doc_term_matrix, num_topics=k, id2word=dictionary)

        # Save model
        lsi_model.save('lsi_model')

        # Print the top 25 words from each topic
        for i in lsi_model.print_topics(num_topics=k, num_words=25):
            print(i)


def find_top_n(cluster_number, top_n, comment_map):
    list = []
    for comment in comment_map:
        if comment_map[comment][0] == cluster_number: #Extract only from the cluster specified
            if len(list) < top_n:
                list.append(comment)

                '''
                if comment_map[comment][1] < min_prob:
                    min_prob = comment_map[comment][1]
                    threshold = min_prob
                '''
            else:
                prob = comment_map[comment][1]
                probs = [comment_map[x][1] for x in list]
                threshold = min(probs)
                if prob > threshold:
                    for x in list: #remove smallest probability
                        if comment_map[x][1] <= threshold:
                            #print('REMOVED: ' + str(cluster_number))
                            list.remove(x)
                            break
                    list.append(comment)
    return list

def representative_comments(k, num_comments, model):
    lda_model = load_model(model)

    #for i in range(1,k):
    #    lda_model.show_topic(k, topn=10)
    #print(lda_model.print_topics(num_topics=50, num_words=10))
    '''
    for i in lda_model.print_topics(num_topics=k, num_words=15):
        print(i)
    '''

    num_clusters = k
    top_n = 25
    comments = get_comments(num_comments, model)
    comments_clean = [clean(comment).split() for comment in comments]
    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(comments_clean)

    map = {}

    '''
    for i in range(0, 10):
        print(comments[i])
        print(comments_clean[i])
        print('***************')
        print('***************')
    '''

    #print('getting started')
    #for i in range(0, len(comments)):
    for i in range(0, num_comments):
        if len(comments[i]) > 125:
            continue
        bag_of_words = []
        comment = comments[i]
        comment_clean = clean(comment)

        '''
        for word in comment_clean:
            bag_of_words.append(word)
        '''

        bag_of_words = comment_clean.split()
        #print('bag of words')
        #print('----------')
        #print('----------')
        #print(bag_of_words)

        if len(bag_of_words) > 0: #this if block removes ending punctuation
            last_word = bag_of_words[len(bag_of_words)-1]
            if len(last_word) > 0 and last_word[len(last_word)-1] in set(string.punctuation):
                last_word = last_word[:-1]
                bag_of_words[len(bag_of_words)-1] = last_word

        bow = dictionary.doc2bow(bag_of_words)
        possible_homes = lda_model.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)

        home = -1
        max_probability = 0

        #print(possible_homes)
        for x in possible_homes:
            if x[1] > max_probability:
                max_probability = x[1]
                home = x[0]


        map[comment] = [home, max_probability]

    clusters = [[] for i in range(0,num_clusters)]

    for item in map:
        clusters[map[item][0]].append(item)


    #print(map)
    #print('----------')
    #print(clusters)


    #print_top_n(k=k,map=map)

    return map #links comment to topic


    #return clusters #cluster[i] is list most representative of comments



def print_top_n(k,map, model):
    lda_model = load_model(model)

    all_top_n = []
    for i in range(0, k):
        top_n = find_top_n(cluster_number=i, top_n=15, comment_map=map)
        all_top_n.append(top_n)
        count = 0
        print('*************')
        print('TOPIC NUMBER ' + str(i))
        print(lda_model.print_topic(i, topn=10))
        print('*************\n')

        for j in top_n:
            count = count + 1
            print(str(count) + ': ' + j)

        print('----------------------------')
        print('----------------------------')
        count = 0



def print_topics(num_topics, num_words, model):
    lda_model = load_model(model)

    for i in lda_model.print_topics(num_topics=num_topics, num_words=num_words):
        print(i)

    return


def compute_w2v_affinity_matrix(topic_similarities, words, t_id=0, top_n=50):
    if topic_similarities:
        word2vec_similarities = np.zeros((top_n, top_n))
        lda_model = gensim.models.ldamodel.LdaModel.load('lda_model')

        words = []
        topic = lda_model.show_topic(topicid=t_id, topn=top_n)
        for term in topic:
            words.append(term[0])

    w2v_model = pickle.load(open('word2vec_model', 'rb'))

    not_in_vocab = set()
    sum_of_included = 0
    num_included_in_vocab = 0

    #print()d

    for i in range(0, top_n):
        for j in range(0, top_n):
            if words[i] in w2v_model.wv.vocab and words[j] in w2v_model.wv.vocab: #Double check how I handle this
                word2vec_similarities[i,j] = w2v_model.similarity(words[i], words[j])
                if word2vec_similarities[i,j] < 0: #eleminate negative values
                    word2vec_similarities[i,j] = 0
            else:
                not_in_vocab.add((i,j))
            #print('Progess: ' + str(progress) + '/' + str(num_paraphrases ** 2))
            #progress = progress + 1

    for i in range(0, top_n):
        for j in range(0, top_n):
            if (i,j) not in not_in_vocab:
                sum_of_included = sum_of_included + word2vec_similarities[i,j]
                num_included_in_vocab = num_included_in_vocab + 1

    if num_included_in_vocab > 0:
        avg_included = sum_of_included/num_included_in_vocab
    else:
        avg_included = 0

    for point in not_in_vocab:
        word2vec_similarities[point[0], point[1]] = avg_included

    return word2vec_similarities

def cluster_topics(k, model):
    if model == 'lda':
        lda_model = gensim.models.ldamodel.LdaModel.load('lda_model')
    elif model == 'depression_lda':
        lda_model = gensim.models.ldamodel.LdaModel.load('depression_lda_model')
    elif model == 'womens_health_lda':
        lda_model = gensim.models.ldamodel.LdaModel.load('womens_health_lda_model')
    elif model == 'cannabis_lda':
        lda_model = gensim.models.ldamodel.LdaModel.load('cannabis_lda_model')
    else:
        return

    all_clusters = []
    formatted_output = []

    for i in range(0, k):
        w2v_affinity_matrix = compute_w2v_affinity_matrix(topic_similarities=True, words=[], t_id = i)
        [clusters, k_sil] = spectral_clustering(w2v_affinity_matrix)
        all_clusters.append(clusters)
        formatted = [[] for _ in range(0,k_sil)]

        words = []
        topic = lda_model.show_topic(topicid=i, topn=50)

        for term in topic:
            words.append(term[0])

        for j in range(0, len(words)):
            formatted[clusters[j]].append(words[j])

        formatted_output.append(formatted)

    return [formatted_output, all_clusters]

def print_clustered_topics(model, num_topics):
    [formatted_output, all_clusters] = cluster_topics(num_topics, model)
    for i in range(0, len(formatted_output)):
        print('TOPIC NUMBER: ' + str(i))
        print('-------------------')
        for j in formatted_output[i]:
            print(j)
            print('-------------------')
        print('**********')
        print('**********')
        print('**********')

def find_k_silhouette(distance_matrix):
    k = 2
    max_score = -1

    for i in range(2, min(11, len(distance_matrix))):
        clusters = SpectralClustering(n_clusters=i, affinity='precomputed', n_init=25,
                                      assign_labels='discretize').fit_predict(distance_matrix)
        silouette_score = metrics.silhouette_score(distance_matrix, clusters)
        if silouette_score > max_score:
            max_score = silouette_score
            k = i
    return k

def find_k_pca(distance_matrix):
    k = 1
    X_scaled = preprocessing.scale(distance_matrix)
    pca = PCA()
    pca.fit(X_scaled)
    explained = pca.explained_variance_ratio_.cumsum()
    for i in range(0, len(explained)):
        if explained[i] > 0.75:
            k = i + 1
            break
    return k

def spectral_clustering(distance_matrix, distance_matrix2 = []):
    k_pca = find_k_pca(distance_matrix)
    #k_sil = find_k_silhouette(distance_matrix)

    clusters = SpectralClustering(n_clusters=k_pca, affinity='precomputed', n_init=25,
                                  assign_labels='discretize').fit_predict(distance_matrix)
    return [clusters, k_pca]



def word2vec_topic_clusters(model): #word2vec clistering output
    [formatted_output, all_clusters] = cluster_topics(num_topics, model)

    for i in range(0, len(formatted_output)):
        print('TOPIC NUMBER: ' + str(i))
        print('-------------------')
        for j in formatted_output[i]:
            print(j)
            print('-------------------')
        print('**********')
        print('**********')
        print('**********')

def find_number_of_topics(num_comments, model):
    # Load Reddit comments
    comments = get_comments(num_comments, model)

    # Clean Reddit comments
    comments_clean = [clean(comment).split() for comment in comments]

    words = []

    for comment in comments_clean:
        for word in comment:
            words.append(word)

    affinity_matrix = compute_w2v_affinity_matrix(topic_similarities=True, words=words)

    #compute_w2v_affinity_matrix(topic_similarities, words, t_id = 0, top_n=25):
    k = find_k_pca(affinity_matrix)

    return k

def main():
    print('Hello World!')

def lipoff():
    num_topics = 25
    num_cannabis_comments = 84641
    cannabis_comments = get_cannabis_comments()
    #train_model(num_topics, num_comments=num_cannabis_comments, model='derm_cannabis_lda')(
    print_topics(num_topics=num_topics, num_words=200, model='derm_cannabis_lda')


def full_pipeline(num_topics, model, num_comments=None):
    #get number of comments
    if num_comments == None:
        num_comments = get_num_comments(model)

    print('num_comments: ', num_comments)

    #train_model
    train_model(num_topics, num_comments, model)

    #print model
    print_topics(num_topics, 200, model)

    # print clustered topics
    print_clustered_topics(model, num_topics)

    #get comments most likely to be produced by each topic
    map = representative_comments(num_topics, num_comments, models[2])
    print_top_n(num_topics, map, models[2])


if __name__ == "__main__":

    models = ['lda', 'depression_lda', 'womens_health_lda', 'cannabis_lda']
    model = models[2]

    #lipoff()

    #full_pipeline(25, models[2])

    get_comments(176000, models[0])

    #num_topics = 25
    #num_comments = 176000
    #num_comments_depression = 2201551
    #num_comments_womens_health = 3518

    #print_topics(num_topics, 200, models[2])


    #train_model(num_topics, num_comments_womens_health, models[2])
    #print_clustered_topics(models[2])
    #k = find_number_of_topics(num_comments)
    #print(k)

    #'''
    #map = representative_comments(num_topics, num_comments_womens_health, models[2])
    #print_top_n(num_topics, map, models[2])
    #'''

    #find_number_of_topics(num_comments_depression, 'depression_lda')

    #'''
    #clusters = cluster_topics(num_topics, 'depression_lda')
    #print_clustered_topics(clusters[0], clusters[1], 'depression_lda')
    #'''
