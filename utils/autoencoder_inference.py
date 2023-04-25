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
unique_train_cars = pd.read_csv("./data/dataset.csv").car_id.unique()

def activation(input, type):
  
    if type.lower()=='selu':
        return F.selu(input)
    elif type.lower()=='elu':
        return F.elu(input)
    elif type.lower()=='relu':
        return F.relu(input)
    elif type.lower()=='relu6':
        return F.relu6(input)
    elif type.lower()=='lrelu':
        return F.leaky_relu(input)
    elif type.lower()=='tanh':
        return F.tanh(input)
    elif type.lower()=='sigmoid':
        return F.sigmoid(input)
    elif type.lower()=='swish':
        return F.sigmoid(input)*input
    elif type.lower()=='identity':
        return input
    else:
        raise ValueError("Unknown non-Linearity Type")


class AutoEncoder(nn.Module):

    def __init__(self, layer_sizes, nl_type='selu', is_constrained=True, dp_drop_prob=0.0, last_layer_activations=True):
        """
        layer_sizes = size of each layer in the autoencoder model
        For example: [10000, 1024, 512] will result in:
            - encoder 2 layers: 10000x1024 and 1024x512. Representation layer (z) will be 512
            - decoder 2 layers: 512x1024 and 1024x10000.
        
        nl_type = non-Linearity type (default: 'selu).
        is_constrained = If true then the weights of encoder and decoder are tied.
        dp_drop_prob = Dropout probability.
        last_layer_activations = Whether to apply activation on last decoder layer.
        """

        super(AutoEncoder, self).__init__()

        self.layer_sizes = layer_sizes
        self.nl_type = nl_type
        self.is_constrained = is_constrained
        self.dp_drop_prob = dp_drop_prob
        self.last_layer_activations = last_layer_activations

        if dp_drop_prob>0:
            self.drop = nn.Dropout(dp_drop_prob)

        self._last = len(layer_sizes) - 2

        # Initaialize Weights
        self.encoder_weights = nn.ParameterList( [nn.Parameter(torch.rand(layer_sizes[i+1], layer_sizes[i])) for i in range(len(layer_sizes) - 1)  ] )

        # "Xavier Initialization" ( Understanding the Difficulty in training deep feed forward neural networks - by Glorot, X. & Bengio, Y. )
        # ( Values are sampled from uniform distribution )
        for weights in self.encoder_weights:
            init.xavier_uniform_(weights)

        # Encoder Bias
        self.encoder_bias = nn.ParameterList( [nn.Parameter(torch.zeros(layer_sizes[i+1])) for i in range(len(layer_sizes) - 1) ] )

        reverse_layer_sizes = list(reversed(layer_sizes)) 
        # reversed returns iterator


        # Decoder Weights
        if is_constrained == False:
            self.decoder_weights = nn.ParameterList( [nn.Parameter(torch.rand(reverse_layer_sizes[i+1], reverse_layer_sizes[i])) for i in range(len(reverse_layer_sizes) - 1) ] )

            for weights in self.decoder_weights:
                init.xavier_uniform_(weights)

        self.decoder_bias = nn.ParameterList( [nn.Parameter(torch.zeros(reverse_layer_sizes[i+1])) for i in range(len(reverse_layer_sizes) - 1) ] )



    def encode(self,x):
        for i,w in enumerate(self.encoder_weights):
            x = F.linear(input=x, weight = w, bias = self.encoder_bias[i] )
            x = activation(input=x, type=self.nl_type)

        # Apply Dropout on the last layer
        if self.dp_drop_prob > 0:
            x = self.drop(x)

        return x


    def decode(self,x):
        if self.is_constrained == True:
            # Weights are tied
            for i,w in zip(range(len(self.encoder_weights)),list(reversed(self.encoder_weights))):
                x = F.linear(input=x, weight=w.t(), bias = self.decoder_bias[i] )
                x = activation(input=x, type=self.nl_type if i != self._last or self.last_layer_activations else 'identity')

        else:

            for i,w in enumerate(self.decoder_weights):
                x = F.linear(input=x, weight = w, bias = self.decoder_weights[i])
                x = activation(input=x, type=self.nl_type if i != self._last or self.last_layer_activations else 'identity')

        return x

    def forward(self,x):
        # Forward Pass
        return self.decode(self.encode(x))


class AEPredictor:
    def __init__(self, weights_path="./weights/autoencoder.pth"):
        self.model = AutoEncoder(
            layer_sizes=[len(unique_train_cars), 512, 512, 1024], 
            nl_type='selu', 
            is_constrained=True, 
            dp_drop_prob=0.0, 
            last_layer_activations=False
        )
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
    
    def _preprocess_user_interactions(self, user_interactions):
        indexes = [unique_train_cars.tolist().index(car_id) for car_id in user_interactions.car_id if car_id in unique_train_cars]
        user_interactions = torch.zeros(size=(1, len(unique_train_cars)))
        for index in indexes:
            user_interactions[0][int(index)] = 1
        return user_interactions
    
    def predict(self, user_interactions, threshold, top_k = 10):
        user_interactions = self._preprocess_user_interactions(user_interactions)
        # batch = torch.tensor(self._create_batch(user_interactions).values).float()
        pred = torch.sigmoid(self.model(user_interactions)).detach().flatten().tolist()
        pred = sorted(list(enumerate(pred)), key=lambda a: a[0], reverse=True)
        pred = [x[0] for x in pred if x[1] > threshold][:top_k]
        pred = [1 if x > threshold else 0 for x in pred]
        pred = np.array(pred).nonzero()[0].tolist()
        return pred

if __name__ == "__main__":
    ae_pred = AEPredictor()
    interactions = pd.read_csv(interactions_path)
    preds = ae_pred.predict()
