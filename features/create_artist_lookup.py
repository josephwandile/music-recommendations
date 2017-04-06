import numpy as np
from sklearn.cluster import KMeans
import requests as req
import pandas as pd
from collections import defaultdict
import time
from features.util import get_region


def cluster_genres(genres, artists_data):
    start = time.clock()

    genre_map = dict()
    for i, genre in enumerate(genres):
        genre_map[genre] = i

    artists_df = pd.DataFrame(artists_data)
    data = artists_df.set_index('name').groupby('artist').groups

    artist_map = dict()
    for i, key in enumerate(data):
        artist_map[key] = i

    X = np.zeros(shape=(2000, len(genres)))

    for key in data:
        genre_ids = [genre_map[genre] for genre in final_spotify_features[key]['genres']]
        artist_id = artist_map[key]
        for genre_id in genre_ids:
            X[artist_id][genre_id] = 1

    clf = KMeans(n_clusters=20)
    classes = clf.fit_predict(X)

    print(time.clock() - start)

    return artist_map, classes


def extract_spotify_features(artists, profiles_data):
    """
    You'll need to go through the process of getting a token if you want this to run in reasonable time.

    The token below is almost certainly expired.
    """

    start = time.clock()

    errors = dict()
    spotify_features = dict()
    genres = set()

    headers = {
        'Authorization': 'Bearer BQCZEZkYqFh2rUPJOpO2qun51jitevJC1ZGgEL9-cXyFCajDHKYrLkYX95Fp85P68HXA9kFrvWDD_Kepv4R2Xg'}

    artists_df = pd.DataFrame(artists)
    data = artists_df.set_index('name').groupby('artist').groups

    for artist, name in data.iteritems():
        payload = {'type': 'artist', 'limit': 1, 'q': 'artist:"{}"'.format(name[0])}

        resp = req.get('https://api.spotify.com/v1/search', params=payload, headers=headers).json()

        if resp.get('error'):
            errors[artist] = resp.get('message')
            continue

        if not resp['artists']['total']:  # No matches found
            errors[artist] = 'No matches found.'
            continue

        spotify_artist = resp['artists']['items'][0]
        genres.update(spotify_artist['genres'])

        spotify_features[artist] = {
            'genres': spotify_artist['genres'],
            'popularity': spotify_artist['popularity'],
            'followers': spotify_artist['followers']['total'],
        }

    print(time.clock() - start)
    return spotify_features, errors, genres


def extract_artistic_features(data):
    start = time.clock()

    artistic_features = defaultdict(dict)

    users = profiles_data.set_index(['age', 'sex', 'country']).groupby('user').groups
    fan_groups = data.set_index(['user', 'plays']).groupby(['artist']).groups

    for artist, fans in fan_groups.iteritems():

        artistic_features[artist] = defaultdict(int)
        for fan, plays in fans:

            age, sex, cn = users[fan][0]

            # Tally of number of listeners in each major region
            region = get_region(cn)
            if region == 'Antarctica':  # This is the weirdest bug in history
                region = 'Other'
            artistic_features[artist][region] += 1

            # Listeners by gender
            if sex == sex:  # Sex is defined
                artistic_features[artist][sex] += 1
                artistic_features[artist]['gender_count'] += 1

            # To calculate average age of listeners
            if age == age:
                artistic_features[artist]['age_count'] += 1
                artistic_features[artist]['sum_of_ages'] += age

            # Total listeners
            artistic_features[artist]['count'] += 1

            # Total plays
            artistic_features[artist]['total_plays'] += plays

        artistic_features[artist]['average_age'] = artistic_features[artist]['sum_of_ages'] / float(
            artistic_features[artist]['age_count'])
        artistic_features[artist]['prop_female'] = artistic_features[artist]['f'] / float(
            artistic_features[artist]['gender_count'])
        artistic_features[artist]['prop_male'] = artistic_features[artist]['m'] / float(
            artistic_features[artist]['gender_count'])

        for r in ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'Other', 'South America']:
            artistic_features[artist][r] = artistic_features[artist][r] / float(artistic_features[artist]['count'])

    print(time.clock() - start)
    return artistic_features


if __name__ == '__main__':

    artists_data = pd.read_csv('../artists.csv')
    profiles_data = pd.read_csv('../profiles.csv')
    train = pd.read_csv('../train.csv')

    spotify_features, errors, genres = extract_spotify_features(artists_data, profiles_data)
    default_features = {artist: {'genres': [], 'popularity': 0, 'followers': 0} for artist in errors}
    final_spotify_features = default_features.copy()
    final_spotify_features.update(spotify_features)
    artist_map, class_labels = cluster_genres(genres, artists_data)

    artistic_features = extract_artistic_features(train)

    artist_lookup = artistic_features.copy()
    for key in artist_lookup:
        artist_lookup[key].update(spotify_features[key])