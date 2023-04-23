import streamlit as st
from st_clickable_images import clickable_images
import random
import  pandas as pd
import os
from config import *

data = pd.read_csv("data/cars_about.csv").dropna()
centroid_car_ids = [177, 188, 270, 387, 246, 296, 187, 116, 255, 173, 259, 171,  89, 149,  90,  74, 320, 180, 149, 197, 252, 297, 291, 243, 368,  67, 228, 147, 193, 244, 211]
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
                liked_df.to_csv("./data/user_interactions.csv", index=False)

    st.sidebar.write(f"Liked Goods {len(liked_df)}/10")
    progress_bar = st.sidebar.progress((min(len(liked_df) / 10, 1.0)))

def first():
    st.title("First Recomendation Algorithm")
    data = pd.read_csv("./data/user_interactions.csv")
    st.write(data)
    for index in range(len(data)):
        row = data.iloc[index]
        car_name = f"{row.car_model} {row.exteriorColor}".replace("/", " ")
        st.write(f"## {car_name}")
        st.image(os.path.join("images", car_name+".jpg"), width= 720)

        with st.expander("Information"):
            for column_name in data.columns:
                st.write(f"{column_name}: {row[column_name]}")

def second():
    st.title("second")

def third():
    st.title("third")


page_names_to_funcs = {
    "Introduction": intro,
    "first algorithm": first,
    "second algorithm":second,
    "third algorithm": third
}

if __name__ == "__main__":

    demo_name = st.sidebar.selectbox(
        "Choose a demo", 
        page_names_to_funcs.keys(), 
        )
    page_names_to_funcs[demo_name]()
