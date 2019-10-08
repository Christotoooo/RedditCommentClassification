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
import random
from Test import * 


feature_names = []

vectorizer = TfidfVectorizer(ngram_range=(1,1),stop_words='english',norm='l2',sublinear_tf=False)
#vectorizer = CountVectorizer(ngram_range=(1,1),stop_words='english',binary=True)

corpus_train = pd.read_csv("reddit_train.csv",usecols=['comments','subreddits'],delimiter=',',sep='\s*,\s*')
corpus_test = pd.read_csv("reddit_train.csv",usecols=['comments','subreddits'],delimiter=',',sep='\s*,\s*')
#random.shuffle(corpus_train)
#corpus_test = corpus_train["comments"]

#corpus_test = corpus_train[60000:]
corpus_test = corpus_train[60000:]
corpus_train = corpus_train[:60000]



english_words = set(nltk.corpus.words.words()\
                    + nltk.corpus.gutenberg.words()\
                    + nltk.corpus.webtext.words()\
                    + nltk.corpus.nps_chat.words()\
                    + nltk.corpus.brown.words() + nltk.corpus.reuters.words())

#spell = SpellChecker()

# def is_english_word(x):
#     return (spell.correction(x) in english_words)

# important for cleaning non-english tokens
#spell = SpellChecker()
# delete_list = ["aaa" ,"bbb","ccc","ddd","eee","fff","ggg","hhh","iii","jjj","kkk","lll","mmm","nnn","ooo","ppp","qqq","rrr","sss","ttt"\
#                ,"uuu","vvv","www","xxx","yyy","zzz"]

                    
def misspell(x):
    if "aaa" not in x and "bbb" not in x and "ccc" not in x and "ddd" not in x \
    and "eee" not in x and "fff" not in x and "ggg" not in x and "hhh" not in x \
    and "iii" not in x and "jjj" not in x and "kkk" not in x and "lll" not in x and "mmm" not in x and "nnn" not in x \
    and "ooo" not in x and "ppp" not in x and "qqq" not in x and "rrr" not in x and "sss" not in x and "ttt" not in x \
    and "uuu" not in x and "vvv" not in x and "www" not in x and "xxx" not in x and "yyy" not in x and "zzz" not in x \
    and "aa" not in x and "zz" not in x:
        return True
    else:
        return False
    

# delete_list = ["aaa" not in x,"bbb" not in x,"ccc" not in x,"ddd" not in x,"eee" not in x,"fff" not in x,"ggg" not in x,\
#                "hhh" not in x,"iii" not in x,"jjj" not in x,"kkk" not in x,"lll" not in x,"mmm" not in x,"nnn" not in x,\
#                "ooo" not in x,"ppp" not in x,"qqq" not in x,"rrr" not in x,"sss" not in x,"ttt" not in x
#                ,"uuu" not in x,"vvv" not in x,"www" not in x,"xxx" not in x,"yyy" not in x,"zzz" not in x]

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
        if x.isalpha() and x not in stops and len(x)<17 and len(x) > 1 and misspell(x):# and is_english_word(x):
#             if len(x) >10:
#                 text_final.append(PorterStemmer().stem(x))
#             else:
            text_final.append(x)
    
    text = " ".join(text_final)
    return text


to_delete = []

# the major function to process the training dataset
# returns (1) a matrix of all training features x (2) a numpy array of y labels
def preprocess():
    df = corpus_train.copy()
    df['comments'] = df['comments'].map(lambda x: preprocess_text(x))
    y_train = df["subreddits"].to_numpy()
    global vectorizer
    x_train = vectorizer.fit_transform(df['comments'])
    
    print(vectorizer.get_feature_names())
    global feature_names
    feature_names = vectorizer.get_feature_names()
    featname=vectorizer.get_feature_names()
    chi_squared,pval = chi2(x_train, y_train)
    featname = pd.DataFrame(featname)
    chi_2 = pd.DataFrame(chi_squared)
    pval = pd.DataFrame(pval)

    data = pd.concat([featname,chi_2,pval],axis=1)
    data.columns = ["word","chi_squared","pval"]
    data = data.sort_values("pval",axis=0)
    
    global to_delete
    to_delete = list(data.index[15000:])     # 保留10000-15000之间最高
    all_cols = np.arange(x_train.shape[1])
    cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, to_delete)))[0]
    x_train = x_train[:, cols_to_keep]
    return x_train, y_train


# function to process the testing set
# returns the matrix of all testing features
def preprocess_testing():
    df = corpus_test.copy()
    #print(df)
    #df = df['comments']
    df['comments'] = df['comments'].map(lambda x: preprocess_text(x))
    #df = df.map(lambda x: preprocess_text(x))
    #y_train = df["subreddits"].to_numpy()
    global vectorizer
    x_train = vectorizer.transform(df["comments"])
    
    all_cols = np.arange(x_train.shape[1])
    global to_delete
    cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, to_delete)))[0]
    x_train = x_train[:, cols_to_keep]
    
    return x_train
    