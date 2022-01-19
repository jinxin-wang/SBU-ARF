#!env python

import pandas as pd
import numpy as np
import csv as csv
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LogisticRegressionCV

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('cleanedTrain.csv', header=0)        # Load the train file into a dataframe
test_df  = pd.read_csv('cleanedTest.csv',  header=0)        # Load the test file into a dataframe

train_labels = train_df.Survived.values
test_id = test_df.PassengerId.values

clist = ['Pclass', 'Age', 'SibSp', 'Parch', 'Embarked', 'Family_Size', 'Title', 'Title_Mr',
 'Deck', 'Side', 'Fare_Per_Person', 'Gender', 'AgeGenderClass', 'Protocole',
 'Fare', 'AgeClass']

clist = ['Pclass', 'Age', 'Gender', 'Embarked', 'Fare', 'Title_Mr', 'Family_Size', 'Protocole', 'AgeGenderClass']

# clist = ['Pclass', 'Age', 'Gender', 'SibSp', 'Parch', 'Embarked', 'Fare', 'Title_Mr', 'Family_Size', 'Protocole', 'AgeGenderClass']

# clist = ['Pclass', 'Age', 'Parch', 'SibSp', 'Embarked', 'Title', 'Gender', 'Fare']

train_df = train_df[clist]
test_df = test_df[clist]


# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

print 'Training...'
# lr = LogisticRegression(solver='liblinear',penalty='l2')
lr = LogisticRegression(solver='newton-cg')
# lr = LogisticRegression(solver='lbfgs')


lr = lr.fit( train_data, train_labels )

print 'Predicting...'
output = lr.predict(test_data).astype(int)

predictions_file = open("LR.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(test_id, output))
predictions_file.close()
print 'Done.'

