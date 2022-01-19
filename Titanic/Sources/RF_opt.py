#!/usr/bin/python

import sys
import pandas as pd
import numpy as np
import csv as csv
from operator import itemgetter
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from RF_crossval import cv_score
from feature_selection import RandomForestClassifierWithCoef,sort_features

train_df = pd.read_csv('cleanedTrain.csv', header=0)
test_df  = pd.read_csv('cleanedTest.csv',  header=0)

train_df = train_df.drop(['Ticket'],axis=1) 
test_df  = test_df.drop(['Ticket'],axis=1) 

'''
y = train_df.Survived.values
# test_id = test_df.PassengerId.values

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_df = train_df.drop(['Survived','Ticket'],axis=1) 
X  = train_df.values
# tX = test_df.values
'''    

def find_best_estimator(estimator, grid_test, X, y):
    print "Hyperparameter optimization using GridSearchCV..."
    grid_search = GridSearchCV(estimator, grid_test, n_jobs=-1, cv=10)
    rst = grid_search.fit(X, y)
    return rst.best_estimator_

'''
forest = RandomForestClassifier()
rst = find_best_estimator(forest, grid, X, y)
print rst.best_estimator_
'''

def optimize_estimator(feature_range, df):
    feature_slist= sort_features(df,100)
    print "feature list : ", feature_slist 
    y = df.Survived.values
    train_df = df.drop(['Survived'],axis=1)
    history = []
    rf = RandomForestClassifier()
    for i in feature_range:
        grid = { "n_estimators"      : range(200,400,50),
            "criterion"         : ["gini", "entropy"],
            "max_features"      : range(3,i+1,2),
            "max_depth"         : [3],
            "min_samples_split" : range(2,21,2)}

        # select features 
        flist = feature_slist[:i]
        X = train_df[flist].values
        # best estimator
        rf = find_best_estimator(rf, grid, X, y)
        # predict in cross validation
        score = cv_score(rf, X, y)
        print "---------------------------------------------------------------"
        print "iteration : %d, score : %f "%(i,score)
        print "estimator [%d]: "%i, rf
        # history.append([i,rf,score])
    return history

f = int(sys.argv[1])
t = int(sys.argv[2])
optimize_estimator(range(f,t), train_df)
