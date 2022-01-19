#!env python

import pandas as pd
import numpy as np
import csv as csv
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict

'''
# TRAIN DATA
train_df = pd.read_csv('cleanedTrain.csv', header=0)

y = train_df.Survived.values
train_df = train_df.drop(['Survived'],axis=1)
X = train_df.values

forest = RandomForestClassifier()
forest = forest.fit(X,y)
'''

def cv_score(estimator, X, y,numIter=20):
    score = 0.
    for i in xrange(numIter):
        prediction = cross_val_predict(estimator,X,y)
        score += metrics.accuracy_score(y, prediction)
    return score/numIter

# print cv_predict(forest, X, y)
