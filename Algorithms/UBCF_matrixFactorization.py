import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

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

def recommend_movies(prediction_df, userID, movies_df, original_ratings_df, num_recommendations = 5):
    user_row_number = userID - 1
    sorted_user_predictions = predicted_rating_df.iloc[user_row_number].sort_values(ascending = False)
    user_data = original_ratings_df[original_ratings_df.userId == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').sort_values(['rating'], ascending = False))
    print('user {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
                                 merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left', 
                                       left_on = 'movieId', right_on = 'movieId').
                                       rename(columns = {user_row_number:'Predictions'}).
                                       sort_values('Predictions', ascending = False).
                                       iloc[:num_recommendations, :-1])
    return user_full, recommendations

already_rated, predictions = recommend_movies(predicted_rating_df, 2, movies_df, ratings_df, 10)
predictions