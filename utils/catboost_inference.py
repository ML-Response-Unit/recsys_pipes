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

class CatPredictor:
    def __init__(self, catboost_path="../weights/catboost"):
        self.catboost = CatBoostClassifier().load_model(catboost_path)
        self.N_POSITIVE = 5
        
        
    def _preprocess_user_interactions(self, user_interactions):
        encoded_cars = encoded_cars_data[encoded_cars_data['car_id'].isin(user_interactions.car_id)].drop(columns=["car_id"])
        encoded_cars = encoded_cars.sample(self.N_POSITIVE)
        current_interactions = pd.DataFrame(
            dict(
                (f"car_{k}_{column}", [encoded_cars[column].iloc[k]]) for k in range(self.N_POSITIVE) \
                    for column in encoded_cars.columns
            )
        )
        return current_interactions
    
    def _create_batch(self, user_interactions):
        renamed_data = encoded_cars_data.rename(
            columns = dict(
                zip(
                    encoded_cars_data.columns, [f'target_{column}' for column in encoded_cars_data.columns]
                    )
                )
            ).drop(columns=["target_car_id"])
        
        # pd.concat([user_interactions]*len(renamed_data)).join(renamed_data)
        batch = pd.concat(
            [pd.concat([user_interactions]*len(renamed_data), ignore_index=True), renamed_data],
            axis=1
            )

        return batch
    
    def predict(self, user_interactions_path="../data/user_interactions.csv"):
        user_interactions = pd.read_csv(user_interactions_path)
        assert len(user_interactions) >= self.N_POSITIVE
        user_interactions = self._preprocess_user_interactions(user_interactions)
        batch = self._create_batch(user_interactions)
        return self.catboost.predict_proba(batch)

catpred = CatPredictor()
print(catpred.predict())
