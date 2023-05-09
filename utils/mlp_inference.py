import pandas as pd
import torch
from config import * 
import torch.nn as nn
import numpy as np

cars_data = pd.read_csv(car_data_path).dropna()

encoded_cars_data = pd.read_csv(encoded_cars_data_path)
encoded_cars_data[encoded_cars_data.columns[1:]] = encoded_cars_data[encoded_cars_data.columns[1:]] / encoded_cars_data[encoded_cars_data.columns[1:]].max(0)

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(66, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

class MLPPredictor:
    def __init__(self, weights_path="./weights/mlp.pth"):
        self.model = BinaryClassification()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()
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
        
        batch = pd.concat(
            [pd.concat([user_interactions]*len(renamed_data), ignore_index=True), renamed_data],
            axis=1
            )

        return batch
    
    def predict(self, user_interactions, threshold, top_k = 10):
        assert len(user_interactions) >= self.N_POSITIVE
        user_interactions = self._preprocess_user_interactions(user_interactions)
        batch = torch.tensor(self._create_batch(user_interactions).values).float()
        pred = torch.sigmoid(self.model(batch)).detach().flatten().tolist()
        pred = sorted(list(enumerate(pred)), key=lambda a: a[1], reverse=True)
        print(pred)
        pred = [x[0] for x in pred if x[1] > threshold][:top_k]
        return pred


if __name__ == "__main__":
    mlp_pred = MLPPredictor()
    interactions = pd.read_csv(interactions_path)
    preds = mlp_pred.predict()
