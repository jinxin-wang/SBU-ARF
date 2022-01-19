#!env python

import pandas as pd
import numpy as np
import csv as csv
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

def sort_features(df,numIter=50):
    y        = df[["PassengerId","Survived"]].values[:, 1]
    train_df = df.drop(["PassengerId","Survived"],axis=1)
    X        = train_df.values
    feature_list = train_df.columns.values[0::]
    forest = RandomForestClassifierWithCoef()
    selector = RFECV(forest, step=1, cv=5)
    support = np.zeros(len(feature_list))
    
    for i in xrange(numIter):
        s = selector.fit(X, y)
        support += np.array(s.support_)

    return np.array(sorted(zip(feature_list, support), key=operator.itemgetter(1),reverse=True))[:,0]

# train_df = pd.read_csv('cleanedTrain.csv', header=0)        # Load the train file into a dataframe
# print sort_features(train_df)
