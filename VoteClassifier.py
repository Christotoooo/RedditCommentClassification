#!/usr/bin/env python
# coding: utf-8

from collections import Counter 
class VoteClassifier:
    def __init__(self, models, precisions):
        self.models = models
        self.precisions = precisions
    
    def fit(self, x, y):
        for model in self.models:
            model.fit(x,y)
    
    def predict(self, y):
        predictions = []
        for pred in y:
            pre = 0
            p = ''
            results = []
            for num, model in enumerate(self.models):
                r = model.predict(pred)[0]
                results.append(r)
                if pre < self.precisions[num][r]:
                    p = r
            win = self.winner(results)
            if(win == 0):
                predictions.append(p)
            else:
                predictions.append(win)
        return predictions
  
    def winner(self, input): 
        # convert list of candidates into dictionary, output will be likes candidates = {'A':2, 'B':4} 
        votes = Counter(input) 
        # create another dictionary and it's key will be count of votes values will be name of candidates 
        dict = {} 
        for value in votes.values():
            # initialize empty list to each key to insert candidate names having same number of votes  
            dict[value] = [] 
        for (key,value) in votes.items(): 
            dict[value].append(key) 

        # sort keys in descending order to get maximum value of votes 
        maxVote = sorted(dict.keys(),reverse=True)[0] 
        if(maxVote < 2):
            return 0
        # check if more than 1 candidates have same number of votes. If yes, then sort the list first and print first element 
        if len(dict[maxVote])>1: 
            return sorted(dict[maxVote])[0] 
        else: 
            return dict[maxVote][0]
