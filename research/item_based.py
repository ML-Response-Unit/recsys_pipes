import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_preprocess(path:str, df:pd.DataFrame, scaler = StandardScaler()) -> pd.DataFrame:
    data_path = os.path.join(path,'cars_about')
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


def get_item_based_reccomendation(user_path:str, datasets_path='../data/') -> pd.DataFrame:
    
    db_cars = load_and_preprocess(datasets_path)

    transformed_db = pd.read_csv('../data/transformed_dataset.csv')

    user_profile = pd.read_csv(os.path.join(datasets_path, user_path))

    user_dist = [transformed_db.iloc[k].to_numpy() for k in user_profile.loc[user_profile['user_id'] == test_user_id].car_id.values]

    dist_matrix_np = np.stack(user_dist, axis=1)
    