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

train_data = {}
with open('data/train.csv', 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user = row[0]
        artist = row[1]
        plays = row[2]

        if not user in train_data:
            train_data[user] = {}

        train_data[user][artist] = int(plays)

with open('user_lookup.pkl', 'rb') as f:
    user_lookup = pkl.load(f)
with open('artist_lookup.pkl', 'rb') as f:
    artist_lookup = pkl.load(f)

# Compute the global median and per-user median.
plays_array = []
user_medians = {}
for user, user_data in train_data.iteritems():
    user_plays = []
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
        user_plays.append(plays)

    user_medians[user] = np.median(np.array(user_plays))
global_median = np.median(np.array(plays_array))

median_data = []
with open('data/train.csv', 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    count = 0
    next(train_csv, None)
    for row in train_csv:
        median_data.append({})
        user   = row[0]
        artist = row[1]
        median_data[count]['user_id'] = user
        median_data[count]['artist_id'] = artist
        median_data[count]['user_median'] = user_medians[user]
        median_data[count]['user_age'] = user_lookup[user]['age']
        median_data[count]['region_share'] = artist_lookup[artist][user_lookup[user]['region']]
        median_data[count]['total_plays'] = user_lookup[user]['total_plays']
        median_data[count]['age_diff'] = abs(user_lookup[user]['age'] - artist_lookup[artist]['average_age'])
        male = (user_lookup[user]['sex'] == 'm')
        female = (user_lookup[user]['sex'] == 'f')
        m_interaction = 0
        f_interaction = 0
        if male:
            m_interaction = artist_lookup[artist]['prop_male']
        if female:
            f_interaction = artist_lookup[artist]['prop_female']
        median_data[count]['m_interaction'] = m_interaction
        median_data[count]['f_interaction'] = f_interaction
        count += 1

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