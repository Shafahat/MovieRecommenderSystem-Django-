import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv', encoding = 'latin-1')
A_df = ratings_df.pivot_table(index = ['userId'], columns = ['movieId'], values = 'rating', aggfunc = np.max)
A_df = A_df.fillna(0)
A = A_df.as_matrix()
user_rating_mean = np.mean(A, axis = 1)
A_normalized = A - user_rating_mean.reshape(-1, 1)
U, sigma, Vt = svds(A_normalized, k = 50)
sigma = np.diag(sigma)
predicted_rating = np.dot(np.dot(U, sigma), Vt) + user_rating_mean.reshape(-1, 1)
predicted_rating_df = pd.DataFrame(predicted_rating, columns = A_df.columns)
preds_df = np.transpose(predicted_rating_df)
item_similarity = cosine_similarity(preds_df)
item_sim_df = pd.DataFrame(item_similarity, index = preds_df.index, columns = preds_df.index)

def sim_movies_to(movieId):
    count = 1
    movieIndex = movies_df.index[movies_df['movieId'] == movieId]
    print('Similar movies to {} are :'.format(movies_df.loc[movieIndex].title))
    for item in item_sim_df.sort_values(by = movieId, ascending = False).index[1:11]:
        itemIndex = movies_df.index[movies_df['movieId']==item]
        print('No. {} : {}'.format(count, movies_df.loc[itemIndex].title))
        count += 1

sim_movies_to(89745)
sim_movies_to(112138)