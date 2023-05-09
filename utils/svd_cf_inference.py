from scipy.sparse.linalg import svds
from scipy.spatial import distance
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

actions_data = pd.read_csv("./data/said_to_actions.csv").dropna().astype(int)

class SVDRecommender:
    
    def __init__(self, k=20):
        self.k = k
    
    def fit(self, data):
        users = list(set(data[:, 0]))
        cars = list(set(data[:, 1]))
        self.n_users = len(users)
        self.n_cars = len(cars)
        self.user_to_idx = {user: i for i, user in enumerate(users)}
        self.car_to_idx = {car: i for i, car in enumerate(cars)}
        self.user_item_matrix = np.zeros((self.n_users, self.n_cars))
        for row in data:
            user_idx = self.user_to_idx[row[0]]
            car_idx = self.car_to_idx[row[1]]
            self.user_item_matrix[user_idx, car_idx] = row[2]
        U, s, Vt = svds(self.user_item_matrix, k=self.k)
        s_diag_matrix = np.diag(s)
        self.X_pred = np.dot(np.dot(U, s_diag_matrix), Vt)
        
    def recommend(self, user_id, n=10):
        user_idx = self.user_to_idx[user_id]
        pred_ratings = self.X_pred[user_idx, :]
        indxs = pred_ratings.argsort()[-n:][::-1]
        top_cars = [list(self.car_to_idx.keys())[list(self.car_to_idx.values()).index(i)] for i in indxs]
        return top_cars

    def find_nearest_user(self, user_vector):
        scores_dict = []
        for user_index in range(self.user_item_matrix.shape[0]):
            similarity = cosine_similarity(
                [self.user_item_matrix[user_index]], 
                [user_vector], 
                dense_output=True
            )
            scores_dict.append((user_index, similarity[0][0]))
        
        scores_dict = sorted(scores_dict, key=lambda a: a[1], reverse=True)
        return scores_dict[0][0]
    
    def inference(self, user_interactions, top_k):
        """
        Runs inference on a neural collaborative filtering (NCF) model 
        to recommend a list of cars to the user based on their past interactions with cars.

        Args:
            user_interactions (pandas.DataFrame): A DataFrame containing the user's past interactions with cars. 
            top_k (int): The number of cars to recommend.

        Returns:
            List of integers: A list of the top `top_k` car IDs recommended to the user, based on their past interactions.
        """
        user_vector = np.zeros(shape=(actions_data.car_id.nunique()))
        for car_id in user_interactions.car_id:
            user_vector[self.car_to_idx[car_id]] = 1
        closest_user_id = self.find_nearest_user(user_vector)
        # closest_user_id = self.get_user_sparse(user_vector, top_k=1).index.tolist()[0]
        return self.recommend(closest_user_id, top_k)
    
actions_data = pd.read_csv("./data/said_to_actions.csv")
actions_data["interaction"] = 1
actions_pivot_table = pd.pivot_table(actions_data, values='interaction', index='user_id', columns='car_id').fillna(0)
actions_data = pd.melt(actions_pivot_table.reset_index(), id_vars='user_id', value_vars=actions_pivot_table.columns).rename(columns={"value":"interaction"})
data = actions_data.values

recommender = SVDRecommender(k=20)
recommender.fit(data)
