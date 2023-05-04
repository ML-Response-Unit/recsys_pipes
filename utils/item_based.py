import pandas as pd
import os
import numpy as np
from scipy.spatial import distance

def get_item_based_reccomendation(user_profile, datasets_path='./data/',top_k:int =30 ) -> pd.DataFrame:

    transformed_db = pd.read_csv(os.path.join(datasets_path, 'transformed_dataset.csv'))
      
    user_dist = [transformed_db.iloc[int(k)].to_numpy() for k in user_profile.car_id.values]

    dist_matrix = np.stack(user_dist, axis=1)
    
    mean_dist = np.mean(dist_matrix, axis=1)
    
    sorted_matrix = transformed_db.apply(lambda x: distance.cosine(x.values, mean_dist), axis='columns').sort_values()

    return sorted_matrix.iloc[:top_k].index.tolist()
