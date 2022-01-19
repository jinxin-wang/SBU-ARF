import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

train_df = pd.read_csv('cleanedTrain.csv', header=0)
# train_df = train_df.drop(['AgeGenderClass'],axis=1)
# train_df = train_df.drop(['Survived','PassengerId'], axis=1)
train_df = train_df.drop(['PassengerId'], axis=1)
# test_df  = pd.read_csv('cleanedTest.csv',  header=0)
sns.corrplot(train_df)
# sns.corrplot(train_df, sig_tail="upper")

plt.show()
