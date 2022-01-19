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

def family_name(df):
    df['Family_Name'] = df['Name'].map(lambda x: x.split(',')[0])
        
def title(df):
    #creating a title column from name
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']

    title_dict={'Master':1, 'Miss':2, 'Mlle':3,
                'Mrs':4, 'Mr':5, 'Major':6, 'Rev':7,
                'Dr':8, 'Ms':9, 'Col':10,
                'Capt':11, 'Mme':12, 'Countess':13,
                'Don':14, 'Jonkheer':15}

    df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))
    df['Title_Mr'] = df[['Title','Sex']].apply(lambda x: replace_titles(x),axis=1)
    df['Title'] = df['Title'].map(lambda x: title_dict[x])
    
def familySize(df):
    #Creating new family_size column
    df['Family_Size']=df['SibSp']+df['Parch']

def clean1(df):
    family_name(df)    
    familySize(df)
    title(df)
    fare(df)
    return df

def update_family_size(x,df):
    family_size = 0
    for i in df.index:
        if df.at[i,'Family_Name'][0] == x['Family_Name']:
            if df.at[i,'Pclass'][0] == x['Pclass']:
                if df.at[i,'Embarked'][0] == x['Embarked']:
                    if df.at[i,'Family_Size'][0] > 0 :
                        family_size += 1
    return family_size
    
def find_age_by_family(x, df):
    for i in df.index:
        if df.at[i,'Family_Name'][0] == x['Family_Name']:
            if df.at[i,'Pclass'][0] == x['Pclass']:
                if df.at[i,'Embarked'][0] == x['Embarked']:
                    if df.at[i,'Family_Size'][0] == x['Family_Size']:
                        if df.at[i,'Ticket'][0] == x['Ticket']:
                            if not pd.isnull(df.at[i,'Age'][0]):
                                return df.at[i,'Age'][0]

def find_cabin_by_family(x, df):
    for i in df.index:
        if df.at[i,'Family_Name'][0] == x['Family_Name']:
            if df.at[i,'Pclass'][0] == x['Pclass']:
                if df.at[i,'Embarked'][0] == x['Embarked']:
                    if df.at[i,'Ticket'][0] == x['Ticket']:
                        if not pd.isnull(df.at[i,'Cabin'][0]):
                            return df.at[i,'Cabin'][0]

def find_cabin_by_ticket(x,df):
    for i in df.index:
        if not pd.isnull(df.at[i,'Cabin'][0]):
            if cmp(df.at[i,'Ticket'][0],x['Ticket'][0]) <= 0:
                return df.at[i,'Cabin'][0]
    
                        
def find_Embarked(x, df):
    for i in df.index:
        if not pd.isnull(df.at[i,'Name'][0]):
            if cmp(df.at[i,'Ticket'][0],x['Ticket'][0]) <= 0:
                return df.at[i,'Embarked'][0]
    
def tichet_to_number(x):
    if x['Ticket'] == 'LINE':
        return 0
    try:
        tNum = int(x['Ticket'])
    except:
        tNum = int(x['Ticket'].split()[-1])
    return tNum
    
def clean2(df):
    #imputing nan values
    classmeans = df.pivot_table('Fare', rows='Pclass', aggfunc='mean')
    df.Fare = df[['Fare', 'Pclass']].apply(lambda x: classmeans[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1 )
    df.Embarked = df.apply(lambda x : find_Embarked(x, df) if pd.isnull(x['Embarked']) else x['Embarked'], axis=1)
    agemeans = df.pivot_table('Age', rows=['Pclass','Title_Mr'], aggfunc='mean')
    df.Cabin = df.apply(lambda x : find_cabin_by_family(x, df) if pd.isnull(x['Cabin']) and x['Family_Size'] > 0 else x['Cabin'], axis=1)
    # df.Cabin = df.apply(lambda x : find_cabin_by_ticket(x, df) if pd.isnull(x['Cabin']) and x['Family_Size'] > 0 else x['Cabin'], axis=1)
    cabin(df)
    df.Age = df.apply(lambda x : find_age_by_family(x, df) if pd.isnull(x['Age']) and x['Family_Size'] > 0 else x['Age'], axis=1)
    df.Age = df[['Age', 'Pclass','Title_Mr']].apply(lambda x: agemeans[x['Pclass'],x['Title_Mr']] if pd.isnull(x['Age']) else x['Age'], axis=1)
    # Fare per person
    df.Family_Size = df.apply(lambda x : update_family_size(x,df), axis=1)
    df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)
    #Age times class
    def protocol(x):
        if x['Pclass'] < 3:
            if x['Title_Mr']<3 or x['Sex']=='female':
                return 1
        return 0
    # female = 1, Male = 2
    df['Gender'] = df['Sex'].map( {'female': 1, 'male': 2} ).astype(int)
    df['AgeClass'] = df['Age']*df['Pclass']
    df['AgeGenderClass'] = df['Age']*df['Gender']*df['Pclass']
    df['Protocole']= df[['Title_Mr', 'Sex','Pclass']].apply(lambda x: protocol(x), axis=1 )
    df.Ticket = df[['Ticket']].apply(lambda x : tichet_to_number(x), axis=1)
    return df

def convert(df):
    # Embarked from 'C', 'Q', 'S'
    # All missing Embarked -> just make them embark from most common place
    if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
        df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
        
    # convert all Embarked strings to int
    Ports = list(enumerate(np.unique(df['Embarked']))) # determine all values of Embarked,
    Ports_dict = { name : i for i, name in Ports }           # set up a dictionary in the form  Ports : index
    df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int) # Convert all Embark strings to int
    return df
    
def clean(no_bins=0):
    # you'll want to tweak this to conform with your computer's file system
    trainpath = 'train.csv'
    testpath  = 'test.csv'
    
    traindf   = pd.read_csv(trainpath)
    testdf    = pd.read_csv(testpath)

    tlabel_df = traindf[['PassengerId','Survived']]
    traindf = traindf.drop(['Survived'], axis=1) 
    df = pd.concat([testdf,traindf])
    df = clean1(df)
    df = clean2(df)
    df = convert(df)
    # Remove the Name column, Cabin, Ticket
    df  = df.drop(['Name', 'Sex', 'Cabin','Family_Name'], axis=1)
    return df,tlabel_df

df,tlabel = clean()
pd.concat([df[df.PassengerId<=len(tlabel)],tlabel[['Survived']]],axis=1).to_csv('cleanedTrain.csv',index=False)
df[df.PassengerId>len(tlabel)].to_csv('cleanedTest.csv',index=False)

