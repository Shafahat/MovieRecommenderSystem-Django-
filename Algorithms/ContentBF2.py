import numpy as np
import pandas as pd

movies = pd.read_csv('movies.csv', encoding = 'latin-1')
movies = pd.concat([movies, movies['genres'].str.get_dummies(sep = '|')], axis = 1)
del movies['genres']
del movies['(no genres listed)']
selected = [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
X = movies.iloc[:, 2:21]

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors = 5).fit(X)
print(nbrs.kneighbors([selected]))