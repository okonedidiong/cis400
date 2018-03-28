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

from datetime import datetime, timedelta
from dateutil import tz, parser


from test2 import get_comments, clean

'''
def get_comments(num_comments):
    df = pd.read_csv('mydata.csv', sep=',', header=None)
    data = df.values
    comments = data[:num_comments, [0]]
    comments = comments.tolist()
    comments = [comment for sublist in comments for comment in sublist]
    comments = [str(comment) for comment in comments]
    return comments
'''


def get_partitioned_comments(num_comments, mode): #mode = 0 for seasons, mode = 1 for years, 2 = seasons and years
    df = pd.read_csv('mydata.csv', sep=',', header=None)
    data = df.values
    utc_list = data[1:num_comments, [3][0]] #skip the 0th element, which is the header
    date_list = [utc_to_date(x) for x in utc_list]

    comments = data[1:num_comments, [0]] #skip 0th element, which is the header
    comments = comments.tolist()
    comments = [comment for sublist in comments for comment in sublist]
    comments = [str(comment) for comment in comments]

    partitioned_comments = []

    if mode == 0:
        season_list = [get_season(x) for x in date_list]
        partitioned_comments = [[],[],[],[]]

        for c in range(0, len(comments)):
            if season_list[c] == 0:
                partitioned_comments[0].append(comments[c])
            elif season_list[c] == 1:
                partitioned_comments[1].append(comments[c])
            elif season_list[c] == 2:
                partitioned_comments[2].append(comments[c])
            elif season_list[c] == 3:
                partitioned_comments[3].append(comments[c])
            else:
                print('Error: Invalid Season')

        return partitioned_comments

    if mode == 1:
        year_list = [date[2] for date in date_list]
        year_set = set(year_list)

        partitioned_comments = [[] for i in range(len(year_list))]
        all_years = sorted(list(year_set))
        #print('all_years: ', all_years)

        for i in range(0, len(comments)): #iterate through comments
            for j in range(0, len(all_years)): #iterate through all the possible years
                #print('year_set: ', year_set)
                #print(len(partitioned_comments))
                if year_list[i] == all_years[j]:
                    partitioned_comments[j].append(comments[i])

        return partitioned_comments

    if mode == 2:
        season_list = [get_season(x) for x in date_list]
        year_list = [date[2] for date in date_list]
        year_set = set(year_list)
        all_years = sorted(list(year_set))

        partitioned_comments = [[] for i in range(len(year_list) + len(season_list))]

        for i in range(0, len(comments)): #iterate through comments
            for j in range(0, len(all_years)): #iterate through all the possible years
                if season_list[i] == 0:
                        partitioned_comments[0].append(comments[c])
                elif season_list[i] == 1:
                    partitioned_comments[1].append(comments[c])
                elif season_list[i] == 2:
                    partitioned_comments[2].append(comments[c])
                elif season_list[i] == 3:
                    partitioned_comments[3].append(comments[c])
                else:
                    print('Error: Invalid Season')

    return []

def utc_to_date(created_utc):
    utc = float(created_utc)
    parsed_date = datetime.utcfromtimestamp(utc)
    year = parsed_date.year
    month = parsed_date.month
    day = parsed_date.day
    return [day,month,year]

def get_season(date): # [0,1,2,3] = [winter, spring, summer, fall]
    month = date[1]

    if (month >= 1 and month <= 2) or month == 12: #winter
        return 0
    elif month >= 3 and month <= 5: #spring
        return 1
    elif month >= 6 and month <= 9: #summer
        return 2
    elif month >= 9 and month <= 11:  # fall
        return 3
    else: #error
        return -1

def print_topics(num_topics):
    lda_model = gensim.models.ldamodel.LdaModel.load('lda_model')
    for i in lda_model.print_topics(num_topics=num_topics, num_words=50):
        print(i)
    return

def get_all_possible_years(num_comments):
    df = pd.read_csv('mydata.csv', sep=',', header=None)
    data = df.values
    utc_list = data[1:num_comments, [3][0]]  # skip the 0th element
    date_list = [utc_to_date(x) for x in utc_list]

    comments = data[:num_comments, [0]]
    comments = comments.tolist()
    comments = [comment for sublist in comments for comment in sublist]
    comments = [str(comment) for comment in comments]

    year_list = [date[2] for date in date_list]
    year_set = set(year_list)
    ordered_year_list = sorted(list(year_set))

    return ordered_year_list


def get_epochs(num_comments, mode):
    if mode == 0: #seasons
        return 4
    elif mode == 1:
        return len(get_all_possible_years(num_comments))
    else:
        return -1

def get_all_probs(k, num_comments, mode):
    lda_model = gensim.models.ldamodel.LdaModel.load('lda_model')
    num_clusters = k
    epochs = get_epochs(num_comments, mode)
    all_epoch_comments = get_partitioned_comments(num_comments, mode)

    all_epoch_probs = []
    #print('len all epoch comments: ', len(all_epoch_comments))
    #print('epochs!!!: ', epochs)
    #print('epochs:', epochs)
    #print(all_epoch_comments)
    #print('len(all_epoch_comments):', len(all_epoch_comments))
    for i in range(0, epochs):
        #print('i: ', i)
        #print(len(all_epoch_comments[0]))
        comments_clean = [clean(comment).split() for comment in all_epoch_comments[i]]
        # Creating the term dictionary of our courpus, where every unique term is assigned an index.
        dictionary = corpora.Dictionary(comments_clean)
        map = {}
        for j in range(0, len(all_epoch_comments[i])): #skip column title
            if len((all_epoch_comments[i])[j]) > 125:
                continue
            comment = (all_epoch_comments[i])[j]
            comment_clean = clean(comment)
            bag_of_words = comment_clean.split()
            if len(bag_of_words) > 0:  # this if block removes ending punctuation
                last_word = bag_of_words[len(bag_of_words) - 1]
                if len(last_word) > 0 and last_word[len(last_word) - 1] in set(string.punctuation):
                    last_word = last_word[:-1]
                    bag_of_words[len(bag_of_words) - 1] = last_word

            bow = dictionary.doc2bow(bag_of_words)
            possible_homes = lda_model.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None,
                                                           per_word_topics=False)
            home = -1
            max_probability = 0
            for x in possible_homes:
                if x[1] > max_probability:
                    max_probability = x[1]
                    home = x[0]
            map[comment] = [home, max_probability]

        clusters = [[] for q in range(0, num_clusters)]

        for item in map:
            clusters[map[item][0]].append(item)

        sum = 0
        epoch_prob = []

        for c in range(0, len(clusters)):
            sum = sum + len(clusters[c])

        for z in range(0, num_topics):
            epoch_prob.append(len(clusters[z])/sum)

        all_epoch_probs.append(epoch_prob)



    return all_epoch_probs
    #return map  # links comment to topic
    #return clusters  # cluster[i] is list of comments most representative of comments

if __name__ == "__main__":
    num_topics = 25
    num_comments = 176000

    #get_comments_utc(20)

    #get_comments(num_comments)
    #print_topics(num_topics)

    season_probs = get_all_probs(num_topics, num_comments, 0)
    year_probs = get_all_probs(num_topics, num_comments, 1)

for p in season_probs:
    print(p)

print('---------------')
print('---------------')
print('---------------')

for p in year_probs:
    print(p)
