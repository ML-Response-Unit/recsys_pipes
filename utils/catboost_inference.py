import pandas as pd
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import Birch
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.metrics import accuracy_score

cars_data = pd.read_csv("../data/cars_about.csv").dropna()
encoded_cars_data = pd.read_csv("../data/cars_data_prepared.csv")
# cars_data["brand"] = cars_data.car_model.apply(lambda a: a.split()[0])
# # cars_data = cars_data.drop(columns=["car_model", "engine", "car_id"])
# cars_data["price"] = cars_data['price'].apply(lambda a: int(a[1:]))

# cars_data['used_label'] = cars_data["used_label"].apply(lambda a: 1 if a=="Used" else 0)
# # Identify the categorical columns
# car_cat_cols = ['exteriorColor', 'interiorColor', 'drivetrain', 'fuelType', 'transmission', 'brand']

# # Convert the categorical columns to numerical using Label Encoding
# for col in car_cat_cols:
#     le = LabelEncoder()
#     cars_data[col] = le.fit_transform(cars_data[col])

# cars_data.to_csv("../data/cars_data_prepared.csv")



class CatPredictor:
    def __init__(self, catboost_path="../weights/catboost"):
        self.catboost = CatBoostClassifier().load_model(catboost_path)
        self.N_POSITIVE = 5
        
    def _preprocess_user_interactions(self, user_interactions):
        encoded_cars = cars_data[cars_data['car_id'].isin(user_interactions.car_id)]
        print(encoded_cars)
        print()
        pass
        # matrix_features = []
        # features = []
        # current_positive_samples = actions_data.query(f"car_id != {target_car_id}").query("interaction == 1")
        # # print()
        # if len(current_positive_samples) < N_POSITIVE: AssertionError("need cold start")

        # for car_id in current_positive_samples.sample(N_POSITIVE).car_id.to_list():
        #     matrix_features.append(cars_data.iloc[car_id].to_list())
        # matrix_features.append(cars_data.iloc[target_car_id].to_list())
        # matrix_features.append(int(actions_data.loc[actions_data.car_id == target_car_id].query(f"user_id == {user_id}").interaction))
        
        # for item in matrix_features:
        #     if isinstance(item, list):
        #         for elem in item:
        #             features.append(elem)
        #     else:
        #         features.append(item)
        # # print(features)
        # dataset = pd.concat([dataset, pd.DataFrame.from_records([dict(zip(dataset.columns, features))])])
    
    def predict(self, user_interactions_path="../data/user_interactions.csv"):
        user_interactions = pd.read_csv(user_interactions_path)
        assert len(user_interactions) == self.N_POSITIVE
        user_interactions = self._preprocess_user_interactions(user_interactions)
        pass

catpred = CatPredictor()
catpred.predict()