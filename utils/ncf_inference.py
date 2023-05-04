from scipy.spatial import distance
import pandas as pd 
import numpy as np
import torch
import sklearn
import sklearn.metrics

actions_data = pd.read_csv("./data/said_to_actions.csv").dropna().astype(int)

class RecSysModel(torch.nn.Module):

    def __init__(self, num_users, num_cars):
        super().__init__()
        """
        Initializes the RecSysModel.

        Args:
            num_users (int): The number of unique users in the training data.
            num_cars (int): The number of unique cars in the training data.
        """
        self.user_embedding = torch.nn.Embedding(num_users, 32)
        self.cars_embedding = torch.nn.Embedding(num_cars, 32)
        self.out = torch.nn.Linear(64, 1)
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.step_scheduler_after = "epoch"

    def forward(self, user_id, car_id, interaction):
        """
        Performs forward pass through the model.

        Args:
            user_id (torch.Tensor): Tensor of user IDs.
            car_id (torch.Tensor): Tensor of car IDs.
            interaction (torch.Tensor): Tensor of interaction values.

        Returns:
            tuple: Tuple containing model output, loss, and evaluation metrics.
        """
        user_embeds = self.user_embedding(user_id)
        car_embeds = self.cars_embedding(car_id)
        output = torch.cat([user_embeds, car_embeds], dim=-1)
        output = self.out(output)
        
        return output
        


model = RecSysModel(
    num_cars=actions_data.car_id.nunique(),
    num_users=actions_data.user_id.nunique()
)
    
def get_user_sparse(
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

def ncf_inference(user_interactions, top_k = 30):
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
    
    closest_user_id = get_user_sparse(user_interactions, top_k=1).index.tolist()
    for car_id in sorted(actions_data.car_id.unique().tolist())[:-1]:
        test_sample = {
            'user_id': torch.tensor(closest_user_id),
            'car_id': torch.tensor([car_id]),
            'interaction': torch.tensor([1.])
            }
        
        output = model(**test_sample)
        cars_recomendations[car_id] = torch.sigmoid(output).detach().cpu().item()

    ids = [id for id, _ in sorted(cars_recomendations.items(), key=lambda a: a[1])][:top_k]
    return ids