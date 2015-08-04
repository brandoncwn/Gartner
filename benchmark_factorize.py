# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import libraries
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer


dirname = r'C:\Users\bcowen\Documents\Data''\\'
filename = r'2015 Benchmarks ver 05 14 15.xlsx'

# Import data into dataframe from excel spreadsheet
df = pd.read_excel(dirname+filename,sheetname='2015 Op Ex Budget')

# Drop rows with nan types
df2 = df.loc[(df['Gartner Type'].dropna(axis=0).index)]
df2 = df2.loc[(df2['Description'].dropna(axis=0).index)]


#==============================================================================
# #%% Bag of words Approach
#==============================================================================

''' Select a column and for that column, create a feature space with an
independent variable for every word in that column. This will allow us to then
create an additional column for every word that become all of our independent
variables (x1, x2, ..., xn) with a binary indicator of whether that word is 
present in the original row of that dataframe

corpus = [
...     'This is the first document.',
...     'This is the second second document.',
...     'And the third one.',
...     'Is this the first document?',

becomes

array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
       [0, 1, 0, 1, 0, 2, 1, 0, 1],
       [1, 0, 0, 0, 1, 0, 1, 1, 0],
       [0, 1, 1, 1, 0, 0, 1, 0, 1]]...)

'''
# Factorize the descriptions as standardized entries

data_sparse = pd.get_dummies(df2['Description'])

#%% Split into test and train sets
# Create randomized indeces to select train and test sets resetting df2 index
# in place to get an contiguous index array
idxs        = np.random.permutation(df2.reset_index().index)
idxs_train  = idxs[0:np.floor(len(idxs)*4/5)]
idxs_test    = idxs[np.ceil(len(idxs)*4/5):]

# Create training sets
#train_X = data_sparse[0:800]
#train_y = pd.factorize(df2['Gartner Type'][0:800])[0]  # zero index to call values only
#test_X  = data_sparse[801:1029]
#test_y  = pd.factorize(df2['Gartner Type'][801:1029])[0]

train_X = data_sparse.reset_index().loc[idxs_train]
train_y = pd.factorize(df2.reset_index()['Gartner Type'].loc[idxs_train])[0]  # zero index to call values only
test_X  = data_sparse.reset_index().loc[idxs_test]
test_y  = pd.factorize(df2.reset_index()['Gartner Type'].loc[idxs_test])[0]

#==============================================================================
# #%% Predictive Analytics
#==============================================================================

#%% Naive Bayes Classifier
# Import Naive Bayes classifiers for supervised learning
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

# Import 3 Naive Bayes classifiers
bnb = BernoulliNB()
gnb = GaussianNB()
mnb = MultinomialNB()

# Train naive bayes classifiers
bnb.fit(train_X, train_y)
gnb.fit(train_X, train_y) # Only works with dense arrays
mnb.fit(train_X, train_y)

# Score the 3 different Naive Bayes classifiers
bnb_preds = bnb.predict(test_X)
gnb_preds = gnb.predict(test_X)
mnb_preds = mnb.predict(test_X)


#==============================================================================
# # Import metrics for scoring
#==============================================================================

from sklearn.metrics import classification_report

print(classification_report(test_y, bnb_preds))
print(classification_report(test_y, gnb_preds))
print(classification_report(test_y, mnb_preds))

#%% Random Forest
from sklearn.ensemble import RandomForestClassifier

# Initialize a random forest
rc = RandomForestClassifier(n_estimators = 100,
                            max_features = 'auto',
                            max_depth = None,
                            min_samples_split = 2,
                            min_samples_leaf = 1,
                            verbose = 1,
                            warm_start = False,
                            class_weight = None)

# Train the forest
rc.fit_transform(train_X, train_y)

# Score the forest
rc_preds = rc.predict(test_X)

# Asses
print(classification_report(test_y, rc_preds))


