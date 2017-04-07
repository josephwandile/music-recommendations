import pandas as pd
import numpy as np
import cPickle as pickle
import time

artist_lookup = pickle.load(open('artist_lookup.pkl'))
user_lookup = pickle.load(open('user_lookup.pkl'))
test = pd.read_csv('test.csv')


def feature_extraction(training_data, num_examples=None):
    start = time.clock()
    if not num_examples:
        num_examples = len(training_data)

    list_of_features = np.empty(num_examples, dtype=dict)

    example_i = 0
    for _, row in training_data[:num_examples].iterrows():

        features = {}

        # ************************************************
        # Features for the user, independent of the artist
        user_features = user_lookup[row['user']]

        #         features['user_id_{}'.format(get_user_id(row['user']))] = 1

        features['user_sex_{}'.format(user_features['sex'])] = 1

        # Seems unlikely to improve model if included raw
        user_cn = user_features['cn']
        user_region = user_features['region']

        user_age = user_features['age']
        features['user_age'] = user_age

        features['user_avg_plays'] = user_features['average_plays']
        features['user_num_artists'] = user_features['num_artists']
        features['user_median_plays'] = user_features['user_median_plays']
        features['user_avg_log_plays'] = user_features['average_log_plays']

        user_popularity = user_features['average_popularity']
        features['user_avg_popularity'] = user_popularity

        features['user_avg_sub_global_avg'] = user_features['user_avg_sub_global_avg']

        #         features['user_fav_genre_{}'.format(user_features['favorite_genres'].most_common(1)[0][0])] = 1

        # ****************************************
        # Just the artist, independent of the user
        artist_features = artist_lookup[row['artist']]

        #         features['artist_id_{}'.format(get_artist_id(row['artist']))] = 1

        artist_popularity = artist_features['popularity']
        features['artist_spotify_popularity'] = artist_popularity

        features['artist_avg_sub_global_avg'] = artist_features['artist_avg_sub_global_avg']

        artist_age = artist_features['average_age']
        features['artist_average_listener_age'] = artist_age

        prop_male, prop_female, prop_unknown = artist_features['m'], artist_features['f'], artist_features['u']

        artist_num_listeners = artist_features['total_listeners']
        features['artist_num_listeners'] = artist_num_listeners

        features['artist_avg_plays'] = artist_features['avg_plays']
        features['artist_median_plays'] = artist_features['median_plays']
        features['artist_avg_log_plays'] = artist_features['log_average_plays']

        features['artist_genre_{}'.format(artist_features['genre_id'])] = 1

        # **********************************************
        # Features dependent on both the artist and user
        features['share_cn'] = 1 if user_cn in [cn for cn, _ in artist_features['fan_cns'].most_common(5)] else 0
        features['share_prob_user_from_region'] = artist_features['fan_cns'][user_cn] / np.sum(
            artist_features['fan_cns'].values())
        features['share_prob_user_from_country'] = artist_features['fan_regions'][user_region] / np.sum(
            artist_features['fan_regions'].values())

        # TODO: Get percentage of fans in same region, and percentage in same country.

        if abs(user_age - artist_age) < 2:
            features['share_age'] = 1

        features['diff_age'] = abs(user_age - artist_age)

        fav_user_genres = set([genre for genre, _ in user_features['favorite_genres'].most_common(4)])
        artist_genres = set(artist_features['genres'])
        shared_genres = fav_user_genres & artist_genres
        g_count = 0
        for g in shared_genres:
            g_count += user_features['favorite_genres'][g]
        features['share_genre'] = g_count
        # TODO Maybe also encode the actual genres?
        #         features['share_genre'] = 1 if len(shared_genres) > 0 else 0

        if abs(user_popularity - artist_popularity) < 5:
            features['share_popularity'] = 1

        features['diff_popularity'] = abs(user_popularity - artist_popularity)

        features['share_prob_user_listens_by_gender_{}'.format(user_features['sex'])] = artist_features[
            user_features['sex']]

        list_of_features[example_i] = features
        example_i += 1

    print(time.clock() - start)
    return list_of_features


feature_list = feature_extraction(test)
pickle.dump(feature_list, open('tester.pkl', 'wb'))