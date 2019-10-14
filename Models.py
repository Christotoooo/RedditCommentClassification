#!/usr/bin/env python
# coding: utf-8


import numpy as np
from sklearn import metrics
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


MNB = MultinomialNB(alpha=0.22)
LR = LogisticRegression(solver='saga', multi_class='multinomial', C=3, max_iter=99999)
LSV = LinearSVC(C=0.26)
BNB = BernoulliNB(alpha=0.0001)


Vote_Classifier = VotingClassifier(estimators=[('lr', LR), ('mnb', MNB), ('lsv', LSV), ('bnb', BNB)], voting='hard')


parameters = {
    'alpha': (0.22, 0.24, 0.26, 0.28, 0.3),
}
mnb = GridSearchCV(MultinomialNB(), parameters, cv=5, verbose=1, n_jobs=-1)
res_mnb = mnb.fit(x, y)
print(mnb.best_score_)
print(mnb.cv_results_)


parameters = {
#     'loss' : ['squared_hinge', 'hinge'],
#     'penalty' : ['l1', 'l2'],
    'C' : np.arange(1.9,2.2,0.06),
}

lsv = GridSearchCV(LinearSVC(), parameters, cv=5, verbose=1, n_jobs=-1)
res_lsv = lsv.fit(x, y)
print(lsv.best_score_)
print(lsv.cv_results_)


precision_LR = {
    "AskReddit" : 0.2497992,
    "GlobalOffensive" : 0.65882353,
    "Music" : 0.69868554,
    "Overwatch" : 0.72890295,
    "anime" : 0.61900098,
    "baseball" : 0.68617021,
    "canada" : 0.46875,
    "conspiracy" : 0.42368421,
    "europe" : 0.51862745,
    "funny" : 0.20402685,
    "gameofthrones" : 0.79793814,
    "hockey" : 0.6617357,
    "leagueoflegends" : 0.69733925,
    "movies" : 0.62562563,
    "nba" : 0.68025078,
    "nfl" : 0.65851172,
    "soccer" : 0.65972945,
    "trees" : 0.48766447,
    "worldnews" : 0.35945152,
    "wow" : 0.73996176
}

precision_MNB = {
    'AskReddit' : 0.23241379,
    'GlobalOffensive' : 0.66734694,
    'Music' : 0.63627639,
    'Overwatch' : 0.68761905,
    'anime' : 0.63901979,
    'baseball' : 0.68089648,
    'canada' : 0.46954987,
    'conspiracy' : 0.44176014,
    'europe' : 0.5638191,
    'funny' : 0.23939929,
    'gameofthrones' : 0.82389289,
    'hockey' : 0.70697168,
    'leagueoflegends' : 0.75451647,
    'movies' : 0.56222802,
    'nba' : 0.69620253,
    'nfl' : 0.66832175,
    'soccer' : 0.71195652,
    'trees' : 0.5093572,
    'worldnews' : 0.36391437,
    'wow' : 0.78736209
}

precision_LSV = {
    'AskReddit' : 0.28015564,
    'GlobalOffensive' : 0.63306085,
    'Music' : 0.65229358,
    'Overwatch' : 0.706,
    'anime' : 0.62545455,
    'baseball' : 0.68239921,
    'canada' : 0.48782863,
    'conspiracy' : 0.4203273,
    'europe' : 0.53207547,
    'funny' : 0.22605042,
    'gameofthrones' : 0.74507874,
    'hockey' : 0.65957447,
    'leagueoflegends' : 0.68711656,
    'movies' : 0.62452471,
    'nba' : 0.69896907,
    'nfl' : 0.6401631,
    'soccer' : 0.62996032,
    'trees' : 0.53180212,
    'worldnews' : 0.35064935,
    'wow' : 0.71298819
}

precision_BNB = {
    'AskReddit' : 0.28,
    'GlobalOffensive' : 0.70,
    'Music' : 0.87,
    'Overwatch' : 0.78,
    'anime' : 0.59,
    'baseball' : 0.57,
    'canada' : 0.45,
    'conspiracy' : 0.43,
    'europe' : 0.49,
    'funny' : 0.18,
    'gameofthrones' : 0.91,
    'hockey' : 0.58,
    'leagueoflegends' : 0.68,
    'movies' : 0.52,
    'nba' : 0.57,
    'nfl' : 0.66,
    'soccer' : 0.63,
    'trees' : 0.38,
    'worldnews' : 0.40,
    'wow' : 0.79
}


models = [LR, MNB, LSV, BNB]
precisions = [precision_LR, precision_MNB, precision_LSV, precision_BNB]

VC = VoteClassifier(models, precisions)

model_validate(VC, x, y)

