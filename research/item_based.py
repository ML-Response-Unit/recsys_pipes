import pandas as pd
import os
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial import distance

def load_and_preprocess(path:str, df:pd.DataFrame=None, scaler = StandardScaler()) -> pd.DataFrame:
    data_path = os.path.join(path,'cars_about.csv')
    data = pd.read_csv(data_path).dropna()
    data = data[data.price != "Not Priced"]

    data.price = data.price.map(lambda a: int(a.replace("$", "")))

    # binarize used_label
    data.used_label = data.used_label.apply(lambda a: int(a=="Used"))
    # split brand name from whole name of a car
    data["brand"] = data.car_model.apply(lambda a: a.split()[0])
    # remove cols because they are useful
    cars_global = data['car_id']
    data = data.drop(["car_model", "car_id"], axis=1)

    categotical_cols = [column_name for column_name in data.columns if isinstance(data[column_name].iloc[0], str)]
 
    for column in categotical_cols:
        data[column] = data[column].apply(str.lower)
        
    label_encoders = dict(zip(
        categotical_cols, [LabelEncoder().fit(data[column_name]) for column_name in categotical_cols]
    ))
    for column_name in categotical_cols:
        data[column_name] = label_encoders[column_name].transform(data[column_name])
    
    data['minMPG'] = data['minMPG'].astype('int')

    scaler = scaler.set_output(transform="pandas")
    
    data_norm = scaler.fit_transform(data)

    return data_norm


def get_item_based_reccomendation(user_path:str, datasets_path='../data/',top_k:int =5 ) -> pd.DataFrame:
    
    # db_cars = load_and_preprocess(datasets_path)
    # print(db_cars)

    transformed_db = pd.read_csv('../data/transformed_dataset.csv')

    user_profile = pd.read_csv(os.path.join(datasets_path, user_path))
      
    user_dist = [transformed_db.iloc[int(k)].to_numpy() for k in user_profile.car_id.values]

    dist_matrix = np.stack(user_dist, axis=1)
    
    mean_dist = np.mean(dist_matrix, axis=1)
    
    sorted_matrix = transformed_db.apply(lambda x: distance.cosine(x.values, mean_dist), axis='columns').sort_values()

    return sorted_matrix[:top_k]
    
if __name__ == "__main__":
    recs = get_item_based_reccomendation('../data/liked.csv')
    # print(recs)
    