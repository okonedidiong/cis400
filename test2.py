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


    '''
    for i in range(len(doc)):
        if type(doc[i]) == float:
            print('******')
            print("tpye is float")
            print(doc[i])
            print('******')
    '''

    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    stop_free2 = " ".join([i for i in stop_free.split() if i not in manual_stop_list])
    internet_slang_free = " ".join([i for i in stop_free2.split() if i not in internet_slang_list])
    punc_free = ''.join(ch for ch in internet_slang_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    stop_free = " ".join([i for i in normalized.split() if i not in stop])
    stop_free2 = " ".join([i for i in stop_free.split() if i not in manual_stop_list])
    internet_slang_free = " ".join([i for i in stop_free2.split() if i not in internet_slang_list])
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

def get_comments(num_comments):
    #df=pd.read_csv('reddit_opiates.csv', sep=',',header=None)
    #df = pd.read_csv('reddit_opiates.csv', sep=',', header=None)
    df = pd.read_csv('mydata.csv', sep=',', header=None)
    data = df.values
    comments = data[:num_comments, [0]] #data.csv
    #print(comments[0:100])
    #comments = data[0:50, [1]] #lifestyle
    comments = comments.tolist()
    #comments = [str(i) for i in comments]
    print('len comments: ', len(comments))

    #print(comments)
    comments = [comment for sublist in comments for comment in sublist]
    '''
    print(comments[0])
    print(comments[1])
    print(comments[2])
    comments = [str(i) for i in comments]
    print(comments[0])
    print(comments[1])
    print(comments[2])

    #print(comments)
    '''
    comments = [str(comment) for comment in comments]

    return comments

def get_doc_term_matrix(comments, dictionary):
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(comment) for comment in comments[0:20]]
    #print(doc_term_matrix)
    #print(np.var(doc_term_matrix))
    return doc_term_matrix

def train_model(k, num_comments):
    comments = get_comments(num_comments)
    comments_clean = [clean(comment).split() for comment in comments]
    #df = pd.read_csv('reddit_opiates.csv', sep=',', header=None)
    #data = df.values

    '''
    print(len(comments_clean))
    print(len(comments_clean[0]))
    print(len(data))
    print(len(data[0]))
    '''
    #for i in range(0, num_comments):
    #    comments_clean[i].append(str(data[i, 2]).lower())


    '''
    for i in range(0,500):
        print(comments[i])
        print(comments_clean[i])
        print('******************')
        print('******************')
    return
    '''

    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(comments_clean)
    doc_term_matrix = get_doc_term_matrix(comments_clean,dictionary)
    #print(doc_term_matrix)
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel
    # Running and Training LDA model on the document term matrix.
    lda_model = Lda(doc_term_matrix, num_topics=k, id2word = dictionary, passes=500)
    # Save model
    lda_model.save('lda_model')
    for i in lda_model.print_topics(num_topics=k, num_words=10):
        print(i)
    #'''

def find_top_n(cluster_number, top_n, comment_map):
    list = []
    next_threshold = 0
    threshold = 0
    list_count = 0
    min_prob = 0

    '''
    for comment in comment_map:
        if comment_map[comment][0] == 12:
            print(comment)
    '''

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
                    #print("lol")
                    for x in list: #remove smallest probability
                        #'''
                        if comment_map[x][1] <= threshold:
                            #print('REMOVED: ' + str(cluster_number))
                            list.remove(x)
                            break
                        #'''
                        '''
                        comment_map[comment][1] == threshold
                        list.remove(x)
                        '''
                    list.append(comment)
    return list

def use_model(k, num_comments):
    lda_model = gensim.models.ldamodel.LdaModel.load('lda_model')


    #for i in range(1,k):
    #    lda_model.show_topic(k, topn=10)
    #print(lda_model.print_topics(num_topics=50, num_words=10))
    '''
    for i in lda_model.print_topics(num_topics=k, num_words=15):
        print(i)
    '''

    print('*****')
    x = lda_model.show_topics(num_topics = k, num_words = 15, log=True, formatted=True)
    print(x)
    print('*****')
    return

    num_clusters = k
    top_n = 10
    comments = get_comments(num_comments)
    comments_clean = [clean(comment).split() for comment in comments]
    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(comments_clean)
    #print(lda_model.print_topics(num_topics=3, num_words=5))
    #print(lda_model.print_topics(num_topics=5, num_words=5))

    map = {}

    for i in range(0, len(comments)):
        ''''''
        if len(comments[i]) > 125:
            continue
        bag_of_words = []
        comment = comments[i]
        comment_clean = clean(comment)
        #print('comment: ', comment)

        for word in comment_clean:
            bag_of_words.append(word)

        bag_of_words = comment.split()

        if len(bag_of_words) > 0:
            last_word = bag_of_words[len(bag_of_words)-1]
            if len(last_word) > 0 and last_word[len(last_word)-1] in set(string.punctuation): #remove end punctuation
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

        '''
        if i in range(0,5):
           print('home ' + str(i) + ': ', str(home))
        '''

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
    print_top_n(k=k,map=map)

def print_top_n(k,map):
    lda_model = gensim.models.ldamodel.LdaModel.load('lda_model')
    all_top_n = []
    for i in range(0, k):
        top_n = find_top_n(cluster_number=i, top_n=10, comment_map=map)
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


def compute_w2v_affinity_matrix(t_id, top_n=25):
    w2v_model = pickle.load(open('word2vec_model', 'rb'))
    #print(type(w2v_model.wv.vocab))

    word2vec_similarities = np.zeros((top_n, top_n))
    lda_model = gensim.models.ldamodel.LdaModel.load('lda_model')

    words = []
    #for i in lda_model.print_topics(num_topics=k, num_words=15):
    topic = lda_model.show_topic(topicid=t_id, topn=25)
    #print(topic)
    for term in topic:
        words.append(term[0])
    #print(words)

    not_in_vocab = set()
    sum_of_included = 0
    num_included_in_vocab = 0

    #print()

    for i in range(0, top_n):
        for j in range(0, top_n):
            if words[i] in w2v_model.wv.vocab and words[j] in w2v_model.wv.vocab: #Double check how I handle this
                word2vec_similarities[i,j] = w2v_model.similarity(words[i], words[j])
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

def cluster_topics(k):
    lda_model = gensim.models.ldamodel.LdaModel.load('lda_model')
    all_clusters = []
    formatted_output = []

    for i in range(0, k):
        w2v_affinity_matrix = compute_w2v_affinity_matrix(t_id = i)
        #num_clusters = find_k_pca(w2v_affinity_matrix)
        [clusters, k_sil] = spectral_clustering(w2v_affinity_matrix)
        all_clusters.append(clusters)
        formatted = [[] for x in range(0,k_sil)]
        print('k_sil:', k_sil)


        words = []
        topic = lda_model.show_topic(topicid=i, topn=25)

        for term in topic:
            words.append(term[0])

        print('--------------------')
        #print(spectral_clustering(w2v_affinity_matrix))
        #print('k_sil: ', find_k_silhouette(w2v_affinity_matrix))
        #print(spectral_clustering(w2v_affinity_matrix))
        #print('k_sil: ', find_k_silhouette(w2v_affinity_matrix))


        for j in range(0, len(words)):
            formatted[clusters[j]].append(words[j])

        formatted_output.append(formatted)

    return [formatted_output, all_clusters]

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
    #print(explained)
    for i in range(0, len(explained)):
        if explained[i] > 0.6:
            k = i + 1
            break
    return k

def spectral_clustering(distance_matrix, distance_matrix2 = []):
    #k_pca = find_k_pca(distance_matrix)
    k_sil = find_k_silhouette(distance_matrix)
    #print('k_sil: ', k_sil)
    #print(find_k_silhouette(distance_matrix))
    #print('k_pca: ', k_pca)
    clusters = SpectralClustering(n_clusters=k_sil, affinity='precomputed', n_init=25,
                                  assign_labels='discretize').fit_predict(distance_matrix)
    return [clusters, k_sil]


def main():
    print('Hello World!')


def test():
    print('Hello World!')

if __name__ == "__main__":
    num_topics = 10
    num_comments = 176000
    #train_model(num_topics, num_comments)

    print('******')
    print('training complete')
    print('***********')

    use_model(num_topics, num_comments)
    #print('')

    '''
    [formatted_output, all_clusters] = cluster_topics(num_topics)
    for i in range(0, len(formatted_output)):
        print('TOPIC NUMBER: ' + str(i))
        print('-------------------')
        for j in formatted_output[i]:
            print(j)
            print('-------------------')
        print('**********')
        print('**********')
        print('**********')
    '''