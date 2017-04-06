import pandas as pd
import pickle as pkl
import csv
import sys, os
from scipy.sparse import *
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import *
from sklearn.linear_model import SGDRegressor
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
from sklearn.preprocessing import normalize
print "loading data"
with open('X.pkl','rb') as f:
    training_data = pkl.load(f)
print "loading labels"
with open('Y.pkl','rb') as f:
    training_labels = pkl.load(f)

clf = RandomForestRegressor(n_estimators=500, n_jobs=-1)

def train(X, y, clf):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    print "fitting"
    clf.fit(X_train, y_train)
    print "dumping model"
    with open('RF.pkl', 'wb') as f:
        pkl.dump(clf, f)
    print "predicting"
    preds = clf.predict(X_val)

    from sklearn.metrics import mean_absolute_error
    with open('RFres.txt', 'w') as f:
        f.write("RF MSE: %.4f" % mean_absolute_error(y_val, preds))
    print("RF MSE: %.4f" % mean_absolute_error(y_val, preds))

v = DictVectorizer()
n_train = 100000
td = v.fit_transform(training_data[:n_train])
ty = training_labels[:n_train]
train(td, ty, clf)