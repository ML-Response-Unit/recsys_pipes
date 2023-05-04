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

data = pd.read_csv(car_data_path).dropna()
liked_df = data.iloc[0:0].dropna()

def intro():
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
                liked_df = liked_df.append(start_data.iloc[index], ignore_index = True)
                liked_df.car_id = liked_df.car_id.astype(int)
                liked_df.to_csv(interactions_path, index=False)

    st.sidebar.write(f"Liked Goods {len(liked_df)}/{min_selected_elements}")
    progress_bar = st.sidebar.progress((min(len(liked_df) / min_selected_elements, 1.0)))

def first():
    liked_df = pd.read_csv(interactions_path)
    cat = CatPredictor()      

    if len(liked_df) >= cat.N_POSITIVE:       
        start_data = data[data['car_id'].isin(data.iloc[cat.predict(liked_df, top_k=top_k_recommendations)].car_id)]
        st.title("Catboost")

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

def second():
    liked_df = pd.read_csv(interactions_path)
    mlp = MLPPredictor()      

    if len(liked_df) >= mlp.N_POSITIVE:       
        start_data = data[data['car_id'].isin(data.iloc[mlp.predict(liked_df, 0.6, top_k=top_k_recommendations)].car_id)]
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

def third():
    liked_df = pd.read_csv(interactions_path)
    ae = AEPredictor()      
     
    start_data = data[data['car_id'].isin(data.iloc[ae.predict(liked_df, 0.5, top_k=top_k_recommendations)].car_id)]
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

def fourth():
    liked_df = pd.read_csv(interactions_path)
     
    start_data = data[data['car_id'].isin(data.iloc[get_item_based_reccomendation(liked_df, top_k=top_k_recommendations)].car_id)]
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


page_names_to_funcs = {
    "Cold start": intro,
    "catboost": first,
    "MLP":second,
    "AutoEncoder": third,
    "Item Based": fourth
}

if __name__ == "__main__":

    demo_name = st.sidebar.selectbox(
        "Choose a demo", 
        page_names_to_funcs.keys(), 
        )
    page_names_to_funcs[demo_name]()
