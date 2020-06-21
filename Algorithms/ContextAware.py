import numpy as np
import pandas as pd

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv', encoding = 'latin-1')
ts = ratings['timestamp']
ts = pd.to_datetime(ts, unit='s').dt.hour
movies['hours'] = ts
merged = ratings.merge(movies, left_on = 'movieId', right_on = 'movieId', suffixes = ['_user', ''])
merged = merged[['userId', 'movieId', 'genres', 'hours']]
merged = pd.concat([merged, merged['genres'].str.get_dummies(sep = '|')], axis = 1)
del merged['genres']
del merged['(no genres listed)']

def activeuserprofile(userId):
    userprofile = merged.loc[merged['userId'] == userId]
    del userprofile['userId']
    userprofile = userprofile.groupby(['hours'], as_index = False, sort = True).sum()
    userprofile.iloc[:, 1:20] = userprofile.iloc[:, 1:20].apply(lambda x:(x - np.min(x))/(np.max(x) - np.min(x)), axis = 1)
    return(userprofile)
    
activeuser = activeuserprofile(30)
recommend = pd.read_csv('recommend.csv')
merged = merged.drop_duplicates()
user_pref = recommend.merge(merged, left_on = 'movieId', right_on = 'movieId', suffixes = ['_user', ''])
product = np.dot(user_pref.iloc[:,2:21].as_matrix(), activeuser.iloc[21, 1:20].as_matrix())
preferences = np.stack((user_pref['movieId'], product), axis = -1)
df = pd.DataFrame(preferences, columns = ['movieId', 'preferences'])
result = (df.sort_values(['preferences'], ascending = False)).iloc[0:10, 0]
