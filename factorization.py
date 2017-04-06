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


# Predict via the user-specific median.
# If the user has no data, use the global median.

train_file = 'data/train.csv'
test_file = 'data/test.csv'
soln_file = 'data/user_median.csv'
profile_file = 'data/profiles.csv'
num_train = 50000

# Read in data
def loadUsers():
    from collections import defaultdict
    from random import random
    average_age = 24.51

    users = defaultdict(dict)
    with open(profile_file, 'r') as train_fh:
        train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
        next(train_csv, None)
        rows = 0
        for row in train_csv:
            if rows % 100000 == 0:
                print 'row', rows
            rows += 1
            user = row[0]
            sex = row[1]
            age = row[2]
            country = row[3]
            if sex == "":
                sex = 'm' if random() < .5 else 'f'
            if age == "":
                age = average_age
            users[user]['sex'] = sex
            users[user]['age'] = float(age)
            users[user]['country'] = country
    return users



with open('profiles_data.pkl','rb') as f:
    profile_data = pkl.load(f)

user_dict = loadUsers()

def loadTrain():
    data = []
    y = []
    users=set()
    artists=set()
    with open(train_file, 'r') as train_fh:
        train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
        next(train_csv, None)
        rows = 0
        for row in train_csv:
            if rows % 100000 == 0:
                print 'row', rows
            user = row[0]
            artist = row[1]
            plays = row[2]
            # if int(plays) >= 800:
            #     continue
            rows+=1
            if rows == num_train:
                return (data, np.array(y), users, artists)
            user_info = user_dict[user]
            data.append({"user_id": user, "artist_id": artist,
                         "age": user_info['age'], "sex": user_info['sex'],
                         "country": user_info["country"],
                         "num_plays": profile_data[user]['num_plays'],
                         "num_artists":profile_data[user]['num_artists']}
                        )
            y.append(float(plays))
            users.add(user)
            artists.add(artist)

    return (data, np.array(y), users, artists)

def loadTest():
    y = []
    data = []
    ids = []
    users=set()
    artists=set()
    with open(test_file, 'r') as train_fh:
        train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
        next(train_csv, None)
        rows = 0
        for row in train_csv:
            if rows % 100000 == 0:
                print 'row', rows
            rows+=1

            id = row[0]
            user = row[1]
            artist = row[2]
            if rows <= 5:
                print id, user, artist
            data.append({"user_id": user, "artist_id": artist,
                         "age": profile_data[user]['age'], "sex": profile_data[user]['sex'],
                         "country": profile_data[user]["country"],
                         "num_plays": profile_data[user]['num_plays'],
                         "num_artists": profile_data[user]['num_artists']}
                        )
            users.add(user)
            artists.add(artist)
            ids.append(id)
            y.append(0)

    return (data, ids, users, artists, y)

(data, y, users, artists) = loadTrain()
# # maxy,miny = max(y) * 500000 ,min(y)
# # print maxy, miny
# # y = [(float(play) - miny)/(maxy - miny) for play in y]
#
#
# print "tt split"
X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=42)
#

v = DictVectorizer()
print "train vectorize"
X_train = v.fit_transform(X_train)
print "val vectorize"
X_val = v.transform(X_val)
#
#
print "training FM"
# Build and train a Factorization Machine
fm = RandomForestRegressor(n_estimators=1000, n_jobs=-1)# pylibfm.FM(num_factors=10, num_iter=30, verbose=True, task="regression", initial_learning_rate=0.8, learning_rate_schedule="optimal")
fm.fit(X_train,y_train)
# # print "dumping"
# with open('RF120.pkl', 'rb') as f:
#     fm = pkl.load(f)

print "predicting"
preds = fm.predict(X_val)
# preds = [float(play)*(maxy - miny) + miny for play in preds]
# y_val = [float(play)*(maxy - miny) + miny for play in y_val]
from sklearn.metrics import mean_absolute_error
with open('resRF120.txt', 'w') as f:
    f.write("FM MSE: %.4f" % mean_absolute_error(y_val,preds))
print("FM MSE: %.4f" % mean_absolute_error(y_val,preds))
print 'y_val', y_val[:100]
print 'preds', preds[:100]
#
# print "loading testing"
# (data_test, ids_test, users_test, artists_test, y_test) = loadTest()
# # _, X_test, _, _ = train_test_split(data_test, y_test, test_size=1, random_state=43)
#
#
# print "test vectorize"
# X_test = v.transform(data_test)


# y_test.extend(fm.predict(X_test[c * 10000 : ]))
# print "leny", len(y_test)
# def chunks(l, n):
#     n = max(1, n)
#     return (l[i:i+n] for i in xrange(0, len(l), n))

# y_test = []
# print "test predicting"
# for i,x in enumerate(chunks(data_test, 10000)):
#     if i % 10 == 0:
#         print "predicting", i * 10000
#     y_test.extend(fm.predict(v.transform(x)))
# y_test = fm.predict(X_test)
# print "solution writing"
# with open(soln_file + 'rf120.csv', 'w') as soln_fh:
#     soln_csv = csv.writer(soln_fh,
#                           delimiter=',',
#                           quotechar='"',
#                           quoting=csv.QUOTE_MINIMAL)
#     soln_csv.writerow(['Id', 'plays'])
#     for id, play in zip(ids_test, y_test):
#         soln_csv.writerow([id, play])





