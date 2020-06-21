import operator
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv', encoding = 'latin-1')
merged = ratings.merge(movies, left_on = 'movieId', right_on = 'movieId', sort = True)
merged = merged[['userId', 'title', 'rating']]
movieRatings = merged.pivot_table(index = ['title'], columns = ['userId'], values = 'rating')
movieRatings = movieRatings.fillna(0)
item_similarity = cosine_similarity(movieRatings)
item_sim_df = pd.DataFrame(item_similarity, index = movieRatings.index, columns = movieRatings.index)

def sim_movies_to(title):
    count = 1
    print('Similar movies to {} are :'.format(title))
    for item in item_sim_df.sort_values(by = title, ascending = False).index[1:11]:
        print('No. {} : {}'.format(count, item))
        count += 1

sim_movies_to('Avengers, The (2012)')
sim_movies_to('22 Jump Street (2014)')