import operator
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv', encoding = 'latin-1')
merged = ratings.merge(movies, left_on = 'movieId', right_on = 'movieId', suffixes = ['_user', ''])
merged.rename(columns = {'rating_user':'user_rating'}, inplace = True)
merged = merged[['userId', 'title', 'rating']]
movieRatings = merged.pivot_table(index = ['title'], columns = ['userId'], values = 'rating')
movieRatings = movieRatings.fillna(0)
model_knn = NearestNeighbors(algorithm = 'brute', metric = 'cosine')
model_knn.fit(movieRatings.values)
movie = 690
distances, indices = model_knn.kneighbors(movieRatings.iloc[movie, :].values.reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:'.format(movieRatings.index[movie]))
    else:
        print('{0} : {1}, with distances of {2}:'.format(i, movieRatings.index[indices.flatten()[i]], distances.flatten()[i]))