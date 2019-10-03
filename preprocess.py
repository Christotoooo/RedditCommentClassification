#!/usr/bin/env python
# coding: utf-8

# In[115]:


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
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.corpus import wordnet
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.feature_selection import chi2
from scipy.stats import uniform
from scipy.stats import randint
from spellchecker import SpellChecker #need to install for some computers
from Test import * 


vectorizer = TfidfVectorizer(ngram_range=(1,1),stop_words='english',norm='l2',sublinear_tf=False)
#vectorizer = CountVectorizer(ngram_range=(1,1),stop_words='english')

corpus_train = pd.read_csv("reddit_train.csv",usecols=['comments','subreddits'],delimiter=',')
corpus_test = pd.read_csv("reddit_test.csv",usecols=['comments'],delimiter=',')
english_words = set(nltk.corpus.words.words()) # important for cleaning non-english tokens
spell = SpellChecker()


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
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ',text)
    text = text.split()
    
    # lemmatization
    lemma = nltk.wordnet.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    
#     # stemming
#     text = [PorterStemmer().stem(word) for word in text]
    
    
    text_final = []
    
    # clean all non-English words, numbers, and other weirdos, stopwords
    for x in text:
        #x = spell.correction(x)
        if x.isalpha() and x not in stops and len(x)<20: #and x in english_words:
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
    featname=vectorizer.get_feature_names()
    chi2,pval = chi2(x_train, y_train)
    featname = pd.DataFrame(featname)
    chi2 = pd.DataFrame(chi2)
    pval = pd.DataFrame(pval)

    data = pd.concat([featname,chi2,pval],axis=1)
    data.columns = ["word","chi2","pval"]
    data = data.sort_values("pval",axis=0)

    to_delete = list(data.index[10000:])     # feature数量 保留10000-15000之间acc最高
    all_cols = np.arange(x_train.shape[1])
    cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, to_delete)))[0]
    x_train = x_train[:, cols_to_keep]
    
    return x_train, y_train


# function to process the testing set
# returns the matrix of all testing features
def preprocess_testing():
    df = corpus_test.copy()
    df['comments'] = df['comments'].map(lambda x: preprocess_text(x))
    #y_train = df["subreddits"].to_numpy()
    x_train = vectorizer.fit_transform(df['comments'])
    return x_train
    

# clf = MultinomialNB()
# #x,y = preprocess()

# model_validate(clf,x,y)


# clf = MultinomialNB(alpha=0.15,fit_prior=True)

# model_validate(clf,m,y)


# #######################Stats memo##########################
# No. of tokens: training set / test set

# No operation: 62853 / 41462
    
# spelling-correction: 
    
# remove stopwords: 62721 / 41331
# remove non-English: 20713 / 16111
# remove non-Englsih/remove stopwords: 20587 / 15986
    
# lemma/remove non-English: 21125 / 16518
# stem/remove non-English: 10462 / 8352

    
# lemma/remove non-English/remove stopwords: 21002 / 16396
# stem/remove non-English/remove stopwords: 10359 / 8250

# lemma/stem/remove non-English/remove stopwords: 10323 / 8236


    

