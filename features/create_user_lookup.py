import pandas as pd
import pickle
from collections import Counter, defaultdict
import time
from features.util import get_region


def extract_user_features(data, artist_lookup):
    start = time.clock()

    demographic_lookup = profiles_data.set_index(['age', 'sex', 'country']).groupby('user').groups
    user_lookup = defaultdict(dict)

    playlists = data.set_index(['artist', 'plays']).groupby(['user']).groups
    for user, playlist in playlists.iteritems():

        age, sex, cn = demographic_lookup[user][0]
        num_artists = len(playlist)
        region = get_region(cn)
        if region == 'Antarctica':
            region = 'Other'

        # TODO Make sure age is in reasonable range...
        user_features = {
            'sex': sex if sex == sex else 'u',
            'age': age if age == age else 0,
            'region': region,
            'average_popularity': 0,
            'num_artists': num_artists,
            'total_plays': 0,
        }

        favorite_genres = Counter()
        total_plays = 0
        total_popularity = 0

        for artist, plays in playlist:
            total_plays += plays
            genres = artist_lookup[artist]['genres']
            for genre in genres:
                favorite_genres[genre] += 1

            total_popularity += artist_lookup[artist]['popularity']

        user_features['average_popularity'] = total_popularity / float(num_artists)
        user_features['total_plays'] = total_plays
        user_features['favorite_genres'] = favorite_genres

        user_lookup[user] = user_features

    print(time.clock() - start)
    return user_lookup

if __name__ == '__main__':
    profiles_data = pd.read_csv('../profiles.csv')
    train = pd.read_csv('../train.csv')
    artist_lookup = pickle.load(open('../artist_lookup.pkl'))
    user_lookup = extract_user_features(train, artist_lookup)
    pickle.dump(user_lookup, open('user_lookup.pkl', 'w'))
