#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
import re
import sys
import csv
import time
import tqdm
import nltk
import math
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.corpus import wordnet
from scipy.stats import uniform
from scipy.stats import randint


vectorizer = CountVectorizer()

corpus_train = pd.read_csv("reddit_train.csv",usecols=['comments','subreddits'],delimiter=',')
corpus_test = pd.read_csv("reddit_test.csv",usecols=['comments'],delimiter=',')
english_words = set(nltk.corpus.words.words()) # important for cleaning non-english tokens


# a helper function to process one comment
def preprocess_text(text): 
    text = text.lower().split()
    stops = set(stopwords.words("english"))
#     text = [w for w in text if not w in stops and len(w) >= 3]
    text = " ".join(text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"^https?:\/\/.*[\r\n]*","", text)
    text = text.split()
    
    # lemmatization
    lemma = nltk.wordnet.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    
    # stemming
    text = [PorterStemmer().stem(word) for word in text]
    
    
    text_final = []
    
    # clean all non-English words, numbers, and other weirdos, stopwords
    for x in text:
        if x.isalpha() and x not in stops and x in english_words:
            text_final.append(x)
    
    text = " ".join(text_final)
    return text


# the major function to process the training dataset
# returns (1) a matrix of all training features x (2) a numpy array of y labels
def preprocess():
    df = corpus_train.copy()
    df['comments'] = df['comments'].map(lambda x: preprocess_text(x))
    y_train = df["subreddits"].to_numpy()
    x_train = vectorizer.fit_transform(df['comments'])
    #print(vectorizer.get_feature_names())
    return x_train, y_train

# function to process the testing set
# returns the matrix of all testing features
def preprocess_testing():
    df = corpus_test.copy()
    df['comments'] = df['comments'].map(lambda x: preprocess_text(x))
    #y_train = df["subreddits"].to_numpy()
    x_train = vectorizer.fit_transform(df['comments'])
    return x_train
    


# # In[27]:


# x,y = preprocess()
# print(x.shape)
# y2 = preprocess_testing()
# print(y2.shape)


