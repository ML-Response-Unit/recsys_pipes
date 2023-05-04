from scipy.sparse.linalg import svds
from scipy.spatial import distance
import pandas as pd
import numpy as np
import torch

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
        user_item_matrix = np.zeros((self.n_users, self.n_cars))
        for row in data:
            user_idx = self.user_to_idx[row[0]]
            car_idx = self.car_to_idx[row[1]]
            user_item_matrix[user_idx, car_idx] = row[2]
        U, s, Vt = svds(user_item_matrix, k=self.k)
        s_diag_matrix = np.diag(s)
        self.X_pred = np.dot(np.dot(U, s_diag_matrix), Vt)
        
    def get_user_sparse(
            self,
            user_interactions:pd.DataFrame, 
            actions_data:pd.DataFrame=actions_data, 
            top_k:int=5
        )-> np.ndarray:
        """
        Returns a Pandas DataFrame of the top `top_k` users whose interaction history with cars is most similar to that of the 
        input user interactions, calculated using the Yule, Jaccard, and Hamming distances.
        
        Args:
        - user_interactions: A Pandas DataFrame containing the user interactions for a single user.
        - actions_data: A Pandas DataFrame containing all available user-car interaction data.
        - top_k: An integer value specifying the number of most similar users to return.
        
        Returns:
        A Pandas DataFrame containing the interaction history of the top `top_k` users whose interaction history with cars is 
        most similar to that of the input user interactions, with a column of distance values indicating the similarity of each 
        user to the input user.
        """
        user_data = pd.DataFrame(
            columns = sorted(actions_data.car_id.unique(), key=int), 
            index=actions_data.user_id.unique()
            )   
        
        for user_id in actions_data.user_id:
            current_user_actions = actions_data.query(f"user_id == {user_id}")
            for car_id in current_user_actions.car_id:
                user_data[int(car_id)].iloc[user_id] = 1#random.randint(4, 5) 
        
        user_actions_sparse = user_data.fillna(0)
        test_user = user_interactions.car_id.astype("int").values
        test_user_sparse = np.zeros(user_actions_sparse.shape[1], dtype=np.int8)
        for i in range(test_user_sparse.shape[0]):
            test_user_sparse[i] = 1 if i in test_user else 0
            
        #calculate distances
        distances = np.array([np.mean([distance.yule(user_actions_sparse.iloc[i].values, test_user_sparse),
                        distance.jaccard(user_actions_sparse.iloc[i].values, test_user_sparse),
                        distance.hamming(user_actions_sparse.iloc[i].values, test_user_sparse)])
                for i in range(len(user_actions_sparse))])
        
        #merge and sort by closest neighbour
        user_sparse_dist = pd.DataFrame(distances,columns=['dist']).sort_values(by='dist').merge(user_actions_sparse, right_index=True, left_index=True)
        
        return user_sparse_dist[:top_k]

    def recommend(self, user_id, n=10):
        user_idx = self.user_to_idx[user_id]
        pred_ratings = self.X_pred[user_idx, :]
        indxs = pred_ratings.argsort()[-n:][::-1]
        top_cars = [list(self.car_to_idx.keys())[list(self.car_to_idx.values()).index(i)] for i in indxs]
        return top_cars

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
        cars_recomendations = {}
        closest_user_id = self.get_user_sparse(user_interactions, top_k=1).index.tolist()[0]
        return self.recommend(closest_user_id, top_k)
    
actions_data = pd.read_csv("./data/said_to_actions.csv")
actions_data["interaction"] = 1
actions_pivot_table = pd.pivot_table(actions_data, values='interaction', index='user_id', columns='car_id').fillna(0)
actions_data = pd.melt(actions_pivot_table.reset_index(), id_vars='user_id', value_vars=actions_pivot_table.columns).rename(columns={"value":"interaction"})
data = actions_data.values

recommender = SVDRecommender(k=20)
recommender.fit(data)
