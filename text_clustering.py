import pandas as pd

import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

import textrank


def word_tokenizer(text):
        #tokenizes and stems the text
        tokens = word_tokenize(text)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
        return tokens


def cluster_sentences(sentences, nb_of_clusters=5):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                        stop_words=stopwords.words('english'),
                                        max_df=0.9,
                                        min_df=0.1,
                                        lowercase=True)
        #builds a tf-idf matrix for the sentences
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        kmeans = KMeans(n_clusters=nb_of_clusters)
        kmeans.fit(tfidf_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)
        return dict(clusters)

def get_comments():
    df=pd.read_csv('reddit_opiates.csv', sep=',',header=None)
    data = df.values
    comments = data[0:100, [0]]
    comments = comments.tolist()
    #print(comments[0:50])
    comments = [comment for sublist in comments for comment in sublist]
    for i in range(len(comments)):
        if type(comments[i]) == float:
            comments[i] = str(0)

    #print(comments[0:10])
    return comments

if __name__ == "__main__":

        sentences = get_comments()
        nclusters= 5
        clusters_list = []
        clusters = cluster_sentences(sentences, nclusters)

        for cluster in range(nclusters):
                #print("cluster ",cluster,":")
                cluster_i = []
                for i,sentence in enumerate(clusters[cluster]):
                        #print("\tsentence ",i,": ",sentences[sentence])
                        cluster_i.append(sentences[sentence])
                clusters_list.append(cluster_i)
        '''
        for cluster in clusters_list:
            print(cluster)
            print('**********')
        '''

        summaries = []

        for cluster in clusters_list:
            cluster_joined = ". ".join(cluster)
            # print(comments)
            summary = textrank.extract_sentences(cluster_joined, summary_length=100, clean_sentences=False, language='english')
            summaries.append(summary)

        for summary in summaries:
            print(summary)
            print('***********')



