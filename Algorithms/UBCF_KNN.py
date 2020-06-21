import operator
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv', encoding = 'latin-1')
merged = ratings.merge(movies, left_on = 'movieId', right_on = 'movieId', suffixes = ['_user', ''])
merged.rename(columns = {'rating_user':'user_rating'}, inplace = True)
merged = merged[['userId', 'title', 'rating']]
movieRatings = merged.pivot_table(index = ['userId'], columns = ['title'], values = 'rating')
movieRatings = movieRatings.fillna(0)

model_knn = NearestNeighbors(algorithm = 'brute', metric = 'cosine')
model_knn.fit(movieRatings.values)

user = 2

distances, indices = model_knn.kneighbors(movieRatings.iloc[user-1, :].values.reshape(1, -1), n_neighbors = 6)
best = []
movieRatings = movieRatings.T
for i in indices.flatten():
    if(i != user-1):
        max_score = movieRatings.loc[:, i + 1].max()
        best.append(movieRatings[movieRatings.loc[:, i + 1] == max_score].index.tolist())
    
user_seen_movies = movieRatings[movieRatings.loc[:, user] > 0].index.tolist()
for i in range(len(best)):
    for j in best[i]:
        if(j in user_seen_movies):
            best[i].remove(j)
                
most_common = {}
for i in range(len(best)):
    for j in best[i]:
        if j in most_common:
            most_common[j] += 1
        else:
            most_common[j] = 1
                
sorted_list = sorted(most_common.items(), key = operator.itemgetter(1), reverse = True)
sorted_list[:5]