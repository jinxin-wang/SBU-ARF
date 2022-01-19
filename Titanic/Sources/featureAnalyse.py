#!env python

import pandas as pd
import numpy as np
import csv as csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('cleanedTrain.csv', header=0)        # Load the train file into a dataframe

features_list = train_df.columns.values[1::]
y = train_df[["PassengerId","Survived"]].values[:, 1]
train_df = train_df.drop(["PassengerId","Survived"],axis=1)
X = train_df.values

# Fit a random forest with (mostly) default parameters to determine feature importance
forest = RandomForestClassifier(oob_score=True, n_estimators=100)
forest.fit(X, y)
feature_importance = forest.feature_importances_
print feature_importance

# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())

# A threshold below which to drop features from the final data set. Specifically, this number represents
# the percentage of the most important feature's importance value
fi_threshold = 0

# Get the indexes of all features over the importance threshold
important_idx = np.where(feature_importance > fi_threshold)[0]

# Create a list of all the feature names above the importance threshold
important_features = features_list[important_idx]
print "\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance):\n", \
        important_features

# Get the sorted indexes of important features
sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
print "\nFeatures sorted by importance (DESC):\n", important_features[sorted_idx]

# Adapted from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure()
plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')
plt.yticks(pos, important_features[sorted_idx[::-1]])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.draw()
plt.show()

# Remove non-important features from the feature set, and reorder those remaining
X = X[:, important_idx][:, sorted_idx]
