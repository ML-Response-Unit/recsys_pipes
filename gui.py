import streamlit as st
from st_clickable_images import clickable_images
import random
import  pandas as pd
import os
from config import *
from utils.catboost_inference import CatPredictor
from utils.mlp_inference import MLPPredictor
from utils.autoencoder_inference import AEPredictor
from utils.item_based import get_item_based_reccomendation
from utils.ncf_inference import ncf_inference
from utils.svd_cf_inference import recommender as svd_inference
from utils.collaborative import ItemBasedCollaborativeFiltering

data = pd.read_csv(car_data_path).dropna()
liked_df = data.iloc[0:0].dropna()

def cold_start_page():
    """
    This function displays a set of cars to the user and asks them to add a certain 
    number of cars to their favorites before car recommendations can be made.
    """
    global liked_df
    start_data = data[data['car_id'].isin(centroid_car_ids)]
    st.title("Few steps before you start...")

    for index in range(len(start_data)):
        row = start_data.iloc[index]
        car_name = f"{row.car_model} {row.exteriorColor}".replace("/", " ")
        st.write(f"## {car_name}")
        st.image(os.path.join("images", car_name+".jpg"), width=640)


        col1, col2 = st.columns([3, 1])
        with col1:
            with st.expander("Information"):
                for column_name in start_data.columns:
                    st.write(f"{column_name}: {row[column_name]}")
        with col2:
            if st.checkbox(
                "Add to favorites", 
                key="intro ".join([str(x) for x in row.fillna('').values.tolist()])
                ):
                liked_df = pd.concat([
                    liked_df, 
                    pd.DataFrame(start_data.iloc[index].apply(lambda a: [a]).to_dict())], 
                    ignore_index = True
                )
                liked_df.car_id = liked_df.car_id.astype(int)
                liked_df.to_csv(interactions_path, index=False)

    st.sidebar.write(f"Liked Goods {len(liked_df)}/{min_selected_elements}")
    progress_bar = st.sidebar.progress((min(len(liked_df) / min_selected_elements, 1.0)))

def catboost_page():
    """
    This function uses a trained CatBoost model to make car 
    recommendations based on the user's favorite cars.
    """
    liked_df = pd.read_csv(interactions_path)
    cat = CatPredictor()      

    if len(liked_df) >= cat.N_POSITIVE:       
        pred_ids = cat.predict(liked_df, top_k=top_k_recommendations)
        pred_ids = [i + 1 for i in pred_ids]
        start_data = data.iloc[pred_ids]
        st.title("Catboost")

        for index, row in start_data.iterrows():
            car_name = f"{row.car_model} {row.exteriorColor}".replace("/", " ")
            st.write(f"## {car_name}")
            st.image(os.path.join("images", car_name+".jpg"), width=720)

            with st.expander("Information"):
                for column_name in start_data.columns:
                    st.write(f"{column_name}: {row[column_name]}")

        st.sidebar.write(f"Liked Goods {len(liked_df)}/{min_selected_elements}")
        st.write("\n")
    else:
        st.warning(f"You must add {min_selected_elements} to favorites")

def mlp_page():
    """
    This page uses the MLP algorithm to recommend cars based on the ones you've liked so far. 
    You need to add at least 5 cars to your favorites list before using this page.
    """
    liked_df = pd.read_csv(interactions_path)
    mlp = MLPPredictor()      

    if len(liked_df) >= mlp.N_POSITIVE: 
        pred_ids = mlp.predict(liked_df, 0.6, top_k=top_k_recommendations)      
        pred_ids = [i + 1 for i in pred_ids]
        start_data = data.iloc[pred_ids]
        st.title("MLP")

        for index in range(len(start_data)):
            row = start_data.iloc[index]
            car_name = f"{row.car_model} {row.exteriorColor}".replace("/", " ")
            st.write(f"## {car_name}")
            st.image(os.path.join("images", car_name+".jpg"), width=720)

            with st.expander("Information"):
                for column_name in start_data.columns:
                    st.write(f"{column_name}: {row[column_name]}")

        st.sidebar.write(f"Liked Goods {len(liked_df)}/{min_selected_elements}")
        st.write("\n")
    else:
        st.warning(f"You must add {min_selected_elements} to favorites")

def autoencoder_page():
    """
    This page uses the Autoencoder algorithm to recommend 
    cars based on the ones you've liked so far.
    """
    liked_df = pd.read_csv(interactions_path)
    ae = AEPredictor()      
    pred_ids = ae.predict(liked_df, 0.5, top_k=top_k_recommendations)
    pred_ids = [i + 1 for i in pred_ids]
    start_data = data.iloc[pred_ids]
    st.title("AutoEncoder")

    for index in range(len(start_data)):
        row = start_data.iloc[index]
        car_name = f"{row.car_model} {row.exteriorColor}".replace("/", " ")
        st.write(f"## {car_name}")
        st.image(os.path.join("images", car_name+".jpg"), width=720)

        with st.expander("Information"):
            for column_name in start_data.columns:
                st.write(f"{column_name}: {row[column_name]}")

    st.sidebar.write(f"Liked Goods {len(liked_df)}/{min_selected_elements}")
    st.write("\n")

def item_based_classic_page():
    """
    This page uses the Item-based Collaborative Filtering algorithm 
    to recommend cars based on the ones you've liked so far.
    """
    liked_df = pd.read_csv(interactions_path)
    pred_ids = get_item_based_reccomendation(liked_df, top_k=top_k_recommendations)

    start_data = data[data['car_id'].isin(data.iloc[pred_ids].car_id)]
    st.title("Item Based (classic)")

    for index in range(len(start_data)):
        row = start_data.iloc[index]
        car_name = f"{row.car_model} {row.exteriorColor}".replace("/", " ")
        st.write(f"## {car_name}")
        st.image(os.path.join("images", car_name+".jpg"), width=720)

        with st.expander("Information"):
            for column_name in start_data.columns:
                st.write(f"{column_name}: {row[column_name]}")

    st.sidebar.write(f"Liked Goods {len(liked_df)}/{min_selected_elements}")
    st.write("\n")

def ncf_page():
    """
    This page uses the Neural Collaborative Filtering algorithm 
    to recommend cars based on the ones you've liked so far.
    """
    liked_df = pd.read_csv(interactions_path)
    pred_ids = ncf_inference(liked_df, top_k=top_k_recommendations)
    pred_ids = [i + 1 for i in pred_ids]
    start_data = data[data['car_id'].isin(data.iloc[pred_ids].car_id)]
    st.title("Neural Colaborative Filtering")

    for index in range(len(start_data)):
        row = start_data.iloc[index]
        car_name = f"{row.car_model} {row.exteriorColor}".replace("/", " ")
        st.write(f"## {car_name}")
        st.image(os.path.join("images", car_name+".jpg"), width=720)

        with st.expander("Information"):
            for column_name in start_data.columns:
                st.write(f"{column_name}: {row[column_name]}")

    st.sidebar.write(f"Liked Goods {len(liked_df)}/{min_selected_elements}")
    st.write("\n")

def svd_page():
    """
    This page uses the Neural Collaborative Filtering algorithm 
    to recommend cars based on the ones you've liked so far.
    """
    liked_df = pd.read_csv(interactions_path)
    pred_ids = svd_inference.inference(liked_df, top_k=top_k_recommendations)
    start_data = data[data['car_id'].isin(data.iloc[pred_ids].car_id)]
    st.title("Collaborative Filtering (SVD)")

    for index in range(len(start_data)):
        row = start_data.iloc[index]
        car_name = f"{row.car_model} {row.exteriorColor}".replace("/", " ")
        st.write(f"## {car_name}")
        st.image(os.path.join("images", car_name+".jpg"), width=720)

        with st.expander("Information"):
            for column_name in start_data.columns:
                st.write(f"{column_name}: {row[column_name]}")

    st.sidebar.write(f"Liked Goods {len(liked_df)}/{min_selected_elements}")
    st.write("\n")

def collaborative_item_based_page():
    """
    This page uses the Collaborative Filtering algorithm 
    to recommend cars based on the ones you've liked so far.
    """
    actions = pd.read_csv("./data/said_to_actions_processed.csv")
    actions_pivot_table = pd.pivot_table(actions, values='interaction', index='user_id', columns='car_id').fillna(0)
    transformed_db = pd.read_csv('./data/transformed_dataset.csv').drop('Unnamed: 0', axis='columns')
    item_based_cf = ItemBasedCollaborativeFiltering(list(range(500)), transformed_db, actions_pivot_table)
    users_ratings = item_based_cf.prepare_data()
    item_based_cf.compute_deviations()

    test_user = pd.read_csv("./data/user_interactions.csv")
    user = {int(test_user['car_id'][i]):1.0 for i in range(len(test_user))}
    pred_ids = item_based_cf.recommend(user, top_k_recommendations).index.tolist()
    start_data = data[data['car_id'].isin(data.iloc[pred_ids].car_id)]

    st.title("Collaborative Filtering (Item Based)")

    for index in range(len(start_data)):
        row = start_data.iloc[index]
        car_name = f"{row.car_model} {row.exteriorColor}".replace("/", " ")
        st.write(f"## {car_name}")
        st.image(os.path.join("images", car_name+".jpg"), width=720)

        with st.expander("Information"):
            for column_name in start_data.columns:
                st.write(f"{column_name}: {row[column_name]}")

    st.sidebar.write(f"Liked Goods {len(liked_df)}/{min_selected_elements}")
    st.write("\n")
    
page_names_to_funcs = {
    "Cold start": cold_start_page,
    "Colaborative Filtering (Neural )":ncf_page,
    "Collaborative Filtering (SVD)":svd_page,
    "Collaborative Filtering (Item Based)":collaborative_item_based_page,
    "Item Based (classic)": item_based_classic_page,
    "AutoEncoder": autoencoder_page,
    "catboost": catboost_page,
    "MLP":mlp_page,
}

if __name__ == "__main__":

    demo_name = st.sidebar.selectbox(
        "Choose a demo", 
        page_names_to_funcs.keys(), 
        )
    page_names_to_funcs[demo_name]()
