#!env python

import pandas as pd
import numpy as np
import csv as csv


def gen_answer():
    judge_df = pd.read_csv('titanic3.csv', header=0)
    test_df  = pd.read_csv('test.csv', header=0)
    
    df_size = len(test_df.Name.values)
    answers = []
    ind     = 0
    for i in range(df_size):
        name = test_df.Name.values[i]
        if not name in judge_df.name.values:
            # raise ValueError('[%d] is not found! '%test_df.PassengerId.values[i])
            print '[%d] is not found! '%test_df.PassengerId.values[i]
            name = name.replace('(','')
            name = name.replace(')','')
            name = name.replace('"','')
            for j,n in enumerate(judge_df.name.values):
                hit = 0
                for ns in name.split():
                    if ns in n:
                        hit += 1
                if hit == len(name.split()):
                    print n
                    print test_df.Name.values[i]
                    ind = j
        else:
            ind = np.argmax(judge_df.name.values == test_df.Name.values[i])
        answers.append(judge_df.survived.values[ind])
    return test_df.PassengerId.values, answers

# ids,ans = gen_answer()

def write_csv(ids,ans,fname):
    predictions_file = open(fname, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, ans))
    predictions_file.close()


# write_csv(ids,ans,"answers.csv")

def judge(testFile, ansFile):
    judge_df = pd.read_csv(ansFile, header=0)
    test_df  = pd.read_csv(testFile, header=0)

    if 0 == sum(judge_df.PassengerId.values <> test_df.PassengerId.values):
        print "Score : ", sum(judge_df.Survived.values == test_df.Survived.values)*1./len(test_df.Survived.values)
            
# judge(testFile='LR.csv', ansFile='answers.csv')
