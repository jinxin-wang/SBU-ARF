#!/usr/bin/python

import sys
import pandas as pd
import numpy as np
import csv as csv
import operator
from operator import itemgetter
from sklearn.feature_selection import RFECV
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from RF_crossval import cv_score

train_df = pd.read_csv('cleanedTrain.csv', header=0)
test_df  = pd.read_csv('cleanedTest.csv',  header=0)

train_df = train_df.drop(['Ticket'],axis=1) 
test_df  = test_df.drop(['Ticket'],axis=1) 

def sort_features(df,numIter=50):
    y        = df[["PassengerId","Survived"]].values[:, 1]
    train_df = df.drop(["PassengerId","Survived"],axis=1)
    X        = train_df.values
    feature_list = train_df.columns.values[0::]
    lr       = LogisticRegression()
    selector = RFECV(lr, step=1, cv=5)
    support = np.zeros(len(feature_list))
    for i in xrange(numIter):
        s = selector.fit(X, y)
        support += np.array(s.support_)

    return np.array(sorted(zip(feature_list, support), key=operator.itemgetter(1),reverse=True))[:,0]
def find_best_estimator(estimator, grid_test, X, y):
    print "Hyperparameter optimization using GridSearchCV..."
    grid_search = GridSearchCV(estimator, grid_test, n_jobs=-1, cv=10)
    rst = grid_search.fit(X, y)
    return rst.best_estimator_

def optimize_estimator(feature_range, df):
    feature_slist= sort_features(df,100)
    print "feature list : ", feature_slist
    exit()
    y = df.Survived.values
    train_df = df.drop(['Survived'],axis=1)
    history = []
    rf = LogisticRegression()
    for i in feature_range:
        grid = {"solver"  : ['newton-cg'],# 'lbfgs','liblinear'],
                "max_iter": [100,500]
                # "penalty" : ['l1','l2']
                }

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

f = 4 # int(sys.argv[1])
t = 20 # int(sys.argv[2])
optimize_estimator(range(f,t), train_df)
