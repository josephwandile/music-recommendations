import numpy as np
import pandas as pd
import csv
import sys, os
from scipy.sparse import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDRegressor


# Predict via the user-specific median.
# If the user has no data, use the global median.

train_file = 'data/train.csv'
test_file = 'data/test.csv'
soln_file = 'data/user_median.csv'

# Load the training data.
train_data = dok_matrix((233286,2000))
user_id = {}
artist_id = {}
cur_uid = 0
cur_aid = 0
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user = row[0]
        artist = row[1]
        plays = row[2]

        if not user in user_id:
            user_id[user] = cur_uid
            cur_uid += 1
        if not artist in artist_id:
            artist_id[artist] = cur_aid
            cur_aid += 1

        train_data[user_id[user], artist_id[artist]] = int(plays)



