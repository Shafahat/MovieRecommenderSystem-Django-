import operator
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv', encoding = 'latin-1')
merged = ratings.merge(movies, left_on = 'movieId', right_on = 'movieId', sort = True)
merged = merged[['userId', 'title', 'rating']]
movieRatings = merged.pivot_table(index = ['userId'], columns = ['title'], values = 'rating')
movieRatings = movieRatings.fillna(0)
user_similarity = cosine_similarity(movieRatings)
user_sim_df = pd.DataFrame(user_similarity, index = movieRatings.index, columns = movieRatings.index)
movieRatings = movieRatings.T

def recommendation(user):
    if user not in movieRatings.columns:
        return('No data available on this User')
    
    sim_user = user_sim_df.sort_values(by = user, ascending = False).index[1:11]
    best = []
    
    for i in sim_user:
        max_score = movieRatings.loc[:, i].max()
        best.append(movieRatings[movieRatings.loc[:, i] == max_score].index.tolist())
    
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
    return(sorted_list)
    
recommendation(2)