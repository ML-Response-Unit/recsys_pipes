from sklearn.cluster import Birch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import random

def load_preprocess(path:str ='../data/') -> pd.DataFrame:
        
    data = pd.read_csv(os.path.join(path, "cars_about.csv")).dropna()
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
    
    #encode columns
    for column in categotical_cols:
        data[column] = data[column].apply(str.lower)
        
    label_encoders = dict(zip(
        categotical_cols, [LabelEncoder().fit(data[column_name]) for column_name in categotical_cols]
    ))
    for column_name in categotical_cols:
        data[column_name] = label_encoders[column_name].transform(data[column_name])

    data['minMPG'] = data['minMPG'].astype('int')   

    
    #normalize data
    scaler = StandardScaler().set_output(transform="pandas")

    data_norm = scaler.fit_transform(data)

    return data_norm

def clusters_info(model, transformed_):

    labels = {i: transformed_.iloc[i].argmin() for i in range(len(transformed_))}
    cluster2label = {i:[] for i in range(len(set(labels.values())))}
    for k in labels:
        cluster2label[labels[k]].append(k)
    unique_labels = model.labels_
    centroids = model.subcluster_centers_
    n_clusters = np.unique(unique_labels).size
    center_labels = birch_model.subcluster_labels_
    centroids_ids = [transformed_.loc[transformed_[col] == transformed_[col].min()].index[0] for col in transformed_.columns]
    center_label2id = {center_labels[i]:centroids_ids[i] for i in range(len(center_labels))}
    return center_label2id, center_labels, cluster2label

def sparse_likes(df:pd.DataFrame) -> pd.DataFrame:
    users = []
    num_users = max(df['user_id'])
    # max_car_id = max(df['car_id'])
    max_car_id = 404
    # print(num_users, max_car_id)
    for user_id in range(num_users):
        likes = np.zeros(max_car_id)
        user_ = df.loc[df['user_id'] == user_id]
        for i in range(len(user_)):
            row = df.iloc[i]
            likes[row['car_id']] = 1
        
        users.append(likes)        
    return users

def generate(center_label2id, center_labels, cluster2label):
    
    n_users = 500

    all_centroids = set(list(center_label2id.keys()))

    n_centroids_user = random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_users, [0.17, 0.17, 0.17, 0.13, 0.13, 0.13, 0.05, 0.05])
    n_cars_user = random.choice(list(range(4, 15)), n_users)

    data = {'car_id':[],
            'user_id':[]}
    user_id =0 
    even = False
    # for user_id in range(n_users):
    while not(even):
        #generating dependant variables
        if user_id %100 ==0 and user_id>0:
            generated_df = pd.DataFrame(data)
            

            sparse = sparse_likes(generated_df)
            sparse_df = pd.DataFrame(sparse, columns=list(range(sparse[0].shape[0]))).astype("int")

            even = sparse_df.sum().all()
            print("-"*15, (sparse_df.sum()==0).sum(),"-"*15)
            
        n_centroids = n_centroids_user[user_id]
        
        n_cars = n_cars_user[user_id]
        
        n_noise = random.choice(list(range(0, n_cars//3)), 1)[0]

        
        user_center_ids =random.choice(center_labels, n_centroids, replace=False)
        
        n_clean_cars = n_cars - n_noise - n_centroids
        

        for cluster_id in user_center_ids:
            data['car_id'].append(center_label2id[cluster_id])
            data['user_id'].append(user_id)
                
        n_cars_per_centroid = (n_clean_cars // n_centroids) +1
        '''
        if cars are more then it is possible to choose, enable multi centroid choice
        
        for now: uniform distribution of cars among centroids
        '''
        
        for cluster_id in user_center_ids:
            n_cars_per_centroid = min(len(cluster2label[cluster_id]), n_cars_per_centroid)
            
            ids = random.choice(cluster2label[cluster_id], n_cars_per_centroid, replace=False)
    #         # cluster_id = 
            for id_ in ids:

                data['car_id'].append(id_)
                data['user_id'].append(user_id)

                #change for multi centroid choice
        
        ind_others = np.random.choice(list(range(1, 404)), n_cars_per_centroid+1, replace=False)
        for ind_ in ind_others:

            data['car_id'].append(ind_)
            data['user_id'].append(user_id)
                            
        #noise car generation: pick random diffetent centroid, choose cars from it
        noisy_centroids_label = list(all_centroids - set(user_center_ids))
        
        noise_center = np.random.choice(noisy_centroids_label, 1)[0]
        
            
        noisy_cars_ids = np.random.choice(cluster2label[noise_center], min(n_noise, len(cluster2label[noise_center])), replace=False)
        for car_id in noisy_cars_ids:
            
            data['car_id'].append(car_id)
            data['user_id'].append(user_id)
        
        #test for even car likes distribution
        
        user_id +=1
    return data
if __name__ == '__main__':    
    
    data_norm  = load_preprocess()
    birch_model = Birch(threshold=2.5, n_clusters=24, branching_factor=3).set_output(transform="pandas")    

    transformed = birch_model.fit_transform(data_norm)
    
    center_label2id, center_labels, cluster2label = clusters_info(birch_model, transformed)
    
    data = generate(center_label2id, center_labels, cluster2label)

    
    

