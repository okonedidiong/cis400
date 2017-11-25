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

def get_comments():
    #df=pd.read_csv('reddit_opiates.csv', sep=',',header=None)
    df = pd.read_csv('reddit_opiates.csv', sep=',', header=None)
    data = df.values
    comments = data[0:10, [0]] #opiates
    for i in range(1, len(comments)):
        print(data[i][5])
        if int(data[i][5]) < 1:
            #print('hi')
            #comments = np.delete(comments, i)
    #comments = data[0:50, [1]] #lifestyle
    comments = comments.tolist()
    #print(comments)
    comments = [comment for sublist in comments for comment in sublist]
    #print(comments)

    return comments

def main():
    comments = get_comments()
    print(comments)

if __name__ == "__main__":
    main()