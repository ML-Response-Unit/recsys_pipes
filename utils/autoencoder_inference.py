import pandas as pd
import torch
from config import * 
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

cars_data = pd.read_csv(car_data_path).dropna()

encoded_cars_data = pd.read_csv(encoded_cars_data_path)
encoded_cars_data[encoded_cars_data.columns[1:]] = encoded_cars_data[encoded_cars_data.columns[1:]] / encoded_cars_data[encoded_cars_data.columns[1:]].max(0)
unique_train_cars = pd.read_csv("./data/said_to_actions.csv").car_id.unique()

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

class AEPredictor:
    def __init__(self, weights_path="./weights/autoencoder.pth"):
        self.model = AE(input_shape = len(unique_train_cars))
        self.model.load_state_dict(torch.load(weights_path), map_location="cpu")
        self.model.eval()
    
    def _preprocess_user_interactions(self, user_interactions):
        indexes = [unique_train_cars.tolist().index(car_id) for car_id in user_interactions.car_id if car_id in unique_train_cars]
        user_interactions = torch.zeros(size=(1, len(unique_train_cars)))
        for index in indexes:
            user_interactions[0][int(index)] = 1
        return user_interactions
    
    def predict(self, user_interactions, threshold, top_k = 10):
        user_interactions = self._preprocess_user_interactions(user_interactions)
        pred = torch.sigmoid(self.model(user_interactions)).detach().flatten().tolist()
        pred = sorted(list(enumerate(pred)), key=lambda a: a[1], reverse=True)
        pred = [x[0] for x in pred[:top_k]]
        return pred

if __name__ == "__main__":
    ae_pred = AEPredictor()
    interactions = pd.read_csv(interactions_path)
    preds = ae_pred.predict()
