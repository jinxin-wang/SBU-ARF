#!env python

import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('cleanedTrain.csv', header=0)        # Load the train file into a dataframe
test_df  = pd.read_csv('cleanedTest.csv',  header=0)        # Load the test file into a dataframe

train_labels = train_df.Survived.values
test_id = test_df.PassengerId.values

# clist = ['AgeGenderClass', 'Fare', 'Fare_Per_Person', 'Family_Size', 'Embarked', 'Deck', 'SibSp', 'Title','Gender','Ticket']
# clist = ['AgeClass', 'Fare_Per_Person', 'SibSp', 'Deck','Ticket','Age','Fare','Protocole']
# clist = ['Ticket', 'AgeGenderClass', 'Protocole', 'Fare_Per_Person', 'Fare', 'Title', 'AgeClass', 'Age', 'SibSp', 'Gender', 'Deck', 'Pclass', 'Title_Mr', 'Embarked', 'Parch', 'Family_Size', 'Side']
clist = ['Fare', 'Fare_Per_Person', 'AgeGenderClass', 'Age', 'AgeClass', 'Protocole', 'Title', 'SibSp', 'Gender', 'Pclass', 'Embarked', 'Title_Mr', 'Parch', 'Side', 'Family_Size']

train_df = train_df[clist[:11]]
test_df = test_df[clist[:11]]

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

print 'Training...'
'''
# 6 features include Ticket
# your submission scored 0.78947
forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=3, max_features=6, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

# 10 features include Ticket
# your submission scored 0.79426
forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=3, max_features=7, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

# 13 features include Ticket
# your submission scored 0.79426
forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=3, max_features=10, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

# 11 features include Ticket
# your submission scored 0.79426
forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=4, max_features=8, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

# 8 features exclude Ticket
# your submission scored 0.79904
forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=3, max_features=6, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=15,
            min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

# 11 features exclude Ticket
# your submission scored 0.79904
forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=3, max_features=6, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
'''

# 10 features exclude Ticket, Deck
# your submission scored 0.80383
forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=3, max_features=6, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=15,
            min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

forest = forest.fit( train_data, train_labels )

print 'Training Score...'
print forest.score( train_data, train_labels )

print 'Predicting...'
output = forest.predict(test_data).astype(int)

predictions_file = open("RF.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(test_id, output))
predictions_file.close()
print 'Done.'

