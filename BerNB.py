import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import math
import seaborn as sns
import matplotlib.pyplot as plt

class BerNB(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, trainingDataMatrixX, trainingDataMatrixY):
        trainingDataMatrixX = trainingDataMatrixX.toarray()
        numOfSamples = np.shape(trainingDataMatrixX)[0]
        numOfFeatures = np.shape(trainingDataMatrixX)[1]
        trainingDataMatrixX = (trainingDataMatrixX > 0).astype(np.int_)
        self.weightsX = np.zeros((20, numOfFeatures))
        self.weightsY = np.zeros((20,1))
        self.subclasses = ['hockey', 'nba', 'leagueoflegends','funny','movies','anime','Overwatch','trees','GlobalOffensive','nfl','AskReddit','gameofthrones','worldnews','conspiracy','wow','europe','canada','Music','baseball','soccer']
        for i in range(numOfSamples):
            for j in range(20):
                if trainingDataMatrixY[i] == self.subclasses[j]:
                    self.weightsY[j] = self.weightsY[j] + 1
                    self.weightsX[j] = self.weightsX[j] + trainingDataMatrixX[i]
                    continue
        # Record log_probabilities in weightsX and Y
        for i in range(20):
            for j in range(numOfFeatures):
                self.weightsX[i][j] = (self.weightsX[i][j] + self.alpha) / (self.weightsY[i] + self.alpha)
        for i in range(20):
            self.weightsY[i] = (self.weightsY[i] + self.alpha) / (numOfSamples + self.alpha)
        return

    def predict(self, vaildationDataMatrixX):
        vaildationDataMatrixX = vaildationDataMatrixX.toarray()
        vaildationDataMatrixX = (vaildationDataMatrixX > 0).astype(np.int_)
        numOfSamples = np.shape(vaildationDataMatrixX)[0]
        numOfFeatures = np.shape(vaildationDataMatrixX)[1]
        resultY = []
        for i in range(numOfSamples):
            probs = []
            for n in range(20):
                logProbability = 1
                for j in range(numOfFeatures):
                    if vaildationDataMatrixX[i][j] == 1:
                        logProbability = logProbability * self.weightsX[n][j]
                    else:
                        logProbability = logProbability * (1 - self.weightsX[n][j])
                logProbability = logProbability * self.weightsY[n]
                probs.append(logProbability)
            max_log_probability = max(probs)
            for m in range(20):
                if probs[m] == max_log_probability:
                    resultY.append(self.subclasses[m])
                    break
        return resultY