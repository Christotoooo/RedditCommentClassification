import numpy as np
import string
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold

from StackEnsembler import StackEnsembler
from VoteClassifier import VoteClassifier


def stack_model_validate(models, precisions, X, y, X_setForFinalPrediction):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size= 0.5)

    clf = StackEnsembler(models)
    clf.fit(X_train,Y_train)
    predictions = clf.predict(X_test)
    X_test_1 = np.hstack((X_test, predictions))

    clf_vote = VoteClassifier(models,precisions)
    clf_vote.fit(X_test_1,Y_test)
    stackResult = clf_vote.predict(X_setForFinalPrediction)
    return stackResult
