import numpy as np
import string
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold



class StackEnsembler:
    def __init__(self, models):
        self.models = models

    def fit(self, x, y):
        for model in self.models:
            model.fit(x, y)

    def predict(self, y):
        predictions_list = []
        for model in self.models:
            predictions_list.append(model.predict(y))
        prediction = np.array(predictions_list)
        prediction.transpose()
        return prediction