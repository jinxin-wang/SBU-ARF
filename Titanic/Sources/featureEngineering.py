# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.stats import mode
import string

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    print big_string
    return np.nan

def fare(df):
    # setting silly values to nan
    df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)

def is_odd(x):
    try:
        if int(x[-1])%2==0:
            return 1
        return -1
    except:
        return 0
    
def cabin(df):
    # Special case for cabins as nan may be signal
    df.Cabin = df.Cabin.fillna('Unknown')    
    # Turning cabin number into Deck
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    cabin_dict = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'T':7, 'G':8, 'Unknown':-1}
    df['Deck'] = df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    df['Deck'] = df['Deck'].map(lambda x: cabin_dict[x])
    df['Side'] = df['Cabin'].map(lambda x: is_odd(x))
    
#replacing all titles with mr, mrs, miss, master
title_dict = { 'Master':1, 'Miss':2,'Mr':3, 'Mrs':4 }
def replace_titles(x):
    if x['Title'] in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return title_dict['Mr']
    elif x['Title'] in ['Countess', 'Mme']:
        return title_dict['Mrs']
    elif x['Title'] in ['Mlle', 'Ms']:
        return title_dict['Miss']
    elif x['Title'] =='Dr':
        if x['Sex']=='Male':
            return title_dict['Mr']
        else:
            return title_dict['Mrs']
    else:
        return title_dict[x['Title']]
        
def title(df):
    #creating a title column from name
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']

    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))
    df['Title']=df[['Title','Sex']].apply(lambda x: replace_titles(x),axis=1)
    
def familySize(df):
    #Creating new family_size column
    df['Family_Size']=df['SibSp']+df['Parch']

def clean1(df):
    familySize(df)
    title(df)
    cabin(df)
    fare(df)
    return df
    
def clean2(train, test):
    #imputing nan values
    for df in [train, test]:
        classmeans = df.pivot_table('Fare', rows='Pclass', aggfunc='mean')
        df.Fare = df[['Fare', 'Pclass']].apply(lambda x: classmeans[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1 )
        modeEmbarked = mode(df.Embarked)[0][0]
        df.Embarked = df.Embarked.fillna(modeEmbarked)

    for df in [train,test]:
        agemeans = df.pivot_table('Age', rows='Pclass', aggfunc='mean')
        df.Age = df[['Age', 'Pclass']].apply(lambda x: agemeans[x['Pclass']] if pd.isnull(x['Age']) else x['Age'], axis=1 )
        
    # Fare per person
    for df in [train, test]:
        df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)
        
    #Age times class
    def protocol(x):
        if x['Age']<15 or x['Sex']=='female':
            return 1
        else:
            return 0
            
    for df in [train, test]:
        df['AgeClass'] = df['Age']*df['Pclass']
        df['Protocole']= df[['Age', 'Sex']].apply(lambda x: protocol(x), axis=1 )

    return [train,test]

def convert(train,test):
    for df in [train,test]:
        # female = 0, Male = 1
        df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

        # Embarked from 'C', 'Q', 'S'
        # All missing Embarked -> just make them embark from most common place
        if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
            df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values

        # convert all Embarked strings to int
        Ports = list(enumerate(np.unique(df['Embarked']))) # determine all values of Embarked,
        Ports_dict = { name : i for i, name in Ports }           # set up a dictionary in the form  Ports : index
        df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int) # Convert all Embark strings to int

    return [train,test]
    
def clean(no_bins=0):
    # you'll want to tweak this to conform with your computer's file system
    trainpath = 'train.csv'
    testpath  = 'test.csv'
    traindf   = pd.read_csv(trainpath)
    testdf    = pd.read_csv(testpath)

    traindf = clean1(traindf)
    testdf  = clean1(testdf)
    
    traindf, testdf = clean2(traindf, testdf)
    traindf, testdf = convert(traindf,testdf)
    
    # Remove the Name column, Cabin, Ticket
    traindf = traindf.drop(['Name', 'Ticket', 'Sex','Cabin'], axis=1)
    testdf  = testdf.drop(['Name', 'Ticket', 'Sex','Cabin'], axis=1)
    
    return [traindf, testdf]

traindf, testdf = clean()
traindf.to_csv('cleanedTrain.csv',index=False)
testdf.to_csv('cleanedTest.csv',index=False)
