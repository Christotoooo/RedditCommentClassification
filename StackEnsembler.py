import numpy as np
import string
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from scipy.sparse import csr_matrix


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
        prediction = prediction.transpose()
        for x in range(prediction.shape[0]):
            for y in range(prediction.shape[1]):
                if prediction[x][y] == 'hockey':
                    prediction[x][y] = 1.0
                elif prediction[x][y] == 'nba':
                    prediction[x][y] = 2.0
                elif prediction[x][y] == 'leagueoflegends':
                    prediction[x][y] = 3.0
                elif prediction[x][y] == 'funny':
                    prediction[x][y] = 4.0
                elif prediction[x][y] == 'movies':
                    prediction[x][y] = 5.0
                elif prediction[x][y] == 'anime':
                    prediction[x][y] = 6.0
                elif prediction[x][y] == 'Overwatch':
                    prediction[x][y] = 7.0   
                elif prediction[x][y] == 'trees':
                    prediction[x][y] = 8.0
                elif prediction[x][y] == 'GlobalOffensive':
                    prediction[x][y] = 9.0
                elif prediction[x][y] == 'nfl':
                    prediction[x][y] = 10.0     
                elif prediction[x][y] == 'AskReddit':
                    prediction[x][y] = 11.0     
                elif prediction[x][y] == 'gameofthrones':
                    prediction[x][y] = 12.0     
                elif prediction[x][y] == 'worldnews':
                    prediction[x][y] = 13.0     
                elif prediction[x][y] == 'conspiracy':
                    prediction[x][y] = 14.0     
                elif prediction[x][y] == 'wow':
                    prediction[x][y] = 15.0
                elif prediction[x][y] == 'europe':
                    prediction[x][y] = 16.0
                elif prediction[x][y] == 'canada':
                    prediction[x][y] = 17.0    
                elif prediction[x][y] == 'Music':
                    prediction[x][y] = 18.0    
                elif prediction[x][y] == 'baseball':
                    prediction[x][y] = 19.0    
                elif prediction[x][y] == 'soccer':
                    prediction[x][y] = 20.0    
        print(prediction)
        return prediction.astype(float)