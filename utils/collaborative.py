import plotly.express as px
import sklearn as sk
import pandas as pd
import numpy as np
import random
from scipy.spatial import distance

class ItemBasedCollaborativeFiltering():    
    
    def __init__(self, users, cars, ratings, k=10):
        self.users = users
        # self.users = self.users.reset_index()
        # self.users = self.users.drop(columns=['index'])

        self.cars_db = cars
        
        
        self.ratings = ratings
        self.ratings = self.ratings.reset_index()
        self.ratings = self.ratings.drop(columns=['user_id'])    
        
        self.k = k
        
        self.frequencies = {}
        self.deviations = {}
        
    def norm_dist(self, index):

        return (self.cars_db.iloc[index].mean() - self.cars_db.min().min()) / (self.cars_db.max().max() - self.cars_db.min().min())
    
    def prepare_data(self):
                
        user_indices = list(self.ratings.index.values)

        users_ratings = []
        for user_index in user_indices:
            rated_book_indices = list(self.ratings.iloc[user_index].to_numpy().nonzero()[0])
            users_ratings.append({user_index: dict(self.ratings[self.ratings.columns[rated_book_indices]].iloc[user_index])})
    
        self.users_ratings = users_ratings
        
        return self.users_ratings
        
        
    def compute_deviations(self):
        
        num_users = len(self.users)
        
        for i in range(num_users):
            for ratings in self.users_ratings[i].values():
                
                for item, rating in ratings.items():
                    # print(item, rating)
                    self.frequencies.setdefault(item, {})
                    self.deviations.setdefault(item, {})
                    
                    for (item2, rating2) in ratings.items():    
                        if item != item2:
                            self.frequencies[item].setdefault(item2, 0)
                            self.deviations[item].setdefault(item2, 0.0)
                            self.frequencies[item][item2] += 1
                            #
                            self.deviations[item][item2] += self.norm_dist(item) - self.norm_dist(item2)
            
            for (item, ratings) in self.deviations.items():
                for item2 in ratings:
                    ratings[item2] /= self.frequencies[item][item2]
    
    
    def slope_one_recommend(self, user_ratings):
        recommendations = {}
        frequencies = {}
        
        for (user_item, user_rating) in user_ratings.items():
        
            for (diff_item, diff_ratings) in self.deviations.items():
                if diff_item not in user_ratings and user_item in self.deviations[diff_item]:
                    freq = self.frequencies[diff_item][user_item]
                    recommendations.setdefault(diff_item, 0.0)
                    frequencies.setdefault(diff_item, 0)
                    #
                    recommendations[diff_item] += (diff_ratings[user_item] + user_rating) * freq
                    frequencies[diff_item] += freq
        
        recommendations = [(k, v / frequencies[k]) for (k, v) in recommendations.items()]
        
        recommendations.sort(key=lambda ratings: ratings[1], reverse = True)
        
        return recommendations
    
    
    def recommend(self, user, top_k:int=5):
        user_dists = self.user_sparse(user)
        slope_score = self.slope_one_recommend(user)
        
        liked_ = user_dists.apply(lambda row: row[row == 1.0].index, axis=1)

        
        cars = {}
        for i, v in enumerate(liked_):
            for id_, dist in slope_score:
                
                if id_ in v:
                    cars[id_] = dist*(2 - user_dists.loc[liked_.index[i]].dist)
        result = pd.DataFrame(cars, index=['score']).transpose().sort_values(by='score', ascending=False)
        
        return result[:top_k]


    def user_sparse(self,test_user:dict, top_k:int=5)-> np.ndarray:        
        
        
        # test_user = pd.read_csv(os.path.join(path, "liked.csv")).car_id.astype("int").values

        test_user_sparse = np.zeros(self.ratings.shape[1], dtype=np.int8)
        for i in range(test_user_sparse.shape[0]):
            test_user_sparse[i] = 1 if i in test_user else 0
                                                                
        #calculate distances
        distances = np.array([np.mean([distance.yule(self.ratings.iloc[i].values, test_user_sparse),
                        distance.jaccard(self.ratings.iloc[i].values, test_user_sparse),
                        distance.hamming(self.ratings.iloc[i].values, test_user_sparse)])
                for i in range(len(self.ratings))])
        
        #merge and sort by closest neighbour
        self.user_sparse_dist = pd.DataFrame(distances,columns=['dist']).sort_values(by='dist').merge(self.ratings, right_index=True, left_index=True)
        
        return self.user_sparse_dist
    

if __name__ == "__main__":
    actions = pd.read_csv("../data/said_to_actions_processed.csv")
    actions_pivot_table = pd.pivot_table(actions, values='interaction', index='user_id', columns='car_id').fillna(0)
    transformed_db = pd.read_csv('../data/transformed_dataset.csv').drop('Unnamed: 0', axis='columns')
    item_based_cf = ItemBasedCollaborativeFiltering(list(range(500)), transformed_db, actions_pivot_table)
    users_ratings = item_based_cf.prepare_data()
    item_based_cf.compute_deviations()

    test_user = pd.read_csv("../data/liked.csv")
    user = {int(test_user['car_id'][i]):1.0 for i in range(len(test_user))}
    recommendations = item_based_cf.recommend(user)
