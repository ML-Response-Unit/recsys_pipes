import pandas as pd
from catboost import CatBoostClassifier
from config import * 

cars_data = pd.read_csv(car_data_path).dropna()
encoded_cars_data = pd.read_csv(encoded_cars_data_path)

class CatPredictor:
    def __init__(self, catboost_path="./weights/catboost"):
        self.catboost = CatBoostClassifier().load_model(catboost_path)
        self.N_POSITIVE = 5
        
        
    def _preprocess_user_interactions(self, user_interactions):
        encoded_cars = encoded_cars_data[encoded_cars_data['car_id'].isin(user_interactions.car_id)].drop(columns=["car_id"])
        encoded_cars = encoded_cars.sample(self.N_POSITIVE)

        current_interactions = dict()
        for k in range(self.N_POSITIVE):
            for column in encoded_cars.columns:
                current_interactions[f"car_{k}_{column}"] = [encoded_cars[column].iloc[k]]
        current_interactions = pd.DataFrame(current_interactions)
        return current_interactions
    
    def _create_batch(self, user_interactions):
        renamed_data = encoded_cars_data.rename(
            columns = dict(zip(
            encoded_cars_data.columns, [f'target_{column}' for column in encoded_cars_data.columns])
            )).drop(columns=["target_car_id"])
        
        # pd.concat([user_interactions]*len(renamed_data)).join(renamed_data)
        user_interactions_batch = []
        for _ in range(len(renamed_data)):
            user_interactions_batch.append(self._preprocess_user_interactions(user_interactions))

        batch = pd.concat([pd.concat(user_interactions_batch, ignore_index=True), renamed_data], axis=1)

        return batch
    
    def predict(self, user_interactions, top_k = 20, threshold = 0.5):
        assert len(user_interactions) >= self.N_POSITIVE
        # user_interactions = self._preprocess_user_interactions(user_interactions)
        batch = self._create_batch(user_interactions)
        pred = self.catboost.predict_proba(batch)[:, 1]
        pred = sorted(list(enumerate(pred)), key=lambda a: a[1], reverse=True)
        pred = [x[0] for x in pred[:top_k]]
        return pred



if __name__ == "__main__":
    catpred = CatPredictor()
    preds = catpred.predict()
