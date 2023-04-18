import io
import os

import numpy as np
import requests
import streamlit as st
from PIL import Image

from streamlit_image_select import image_select




st.set_page_config("Demo for streamlit-image-select", "🖼️")

st.title("Выберите интересующие вас авто")
img = image_select(
    label="Select a cat...",
    images=[
        "./images/car_1.jpg",
        "https://bagongkia.github.io/react-image-picker/0759b6e526e3c6d72569894e58329d89.jpg",
        Image.open("./images/car_1.jpg",),
        np.array(Image.open("./images/car_1.jpg",)),
    ],
    captions=["A cat", "Another cat", "Oh look, a cat!", "Guess what, a cat..."],
)

img = image_select(
    label="Select a .",
    images=[
        "./images/car_1.jpg",
        "https://bagongkia.github.io/react-image-picker/0759b6e526e3c6d72569894e58329d89.jpg",
        Image.open("./images/car_1.jpg",),
        np.array(Image.open("./images/car_1.jpg",)),
    ] * 2,
    captions=["A cat", "Another cat", "Oh look, a cat!", "Guess what, a cat..."] * 2,
    use_container_width=True
)

img = image_select(
    label="Select a cat",
    images=[
        "./images/car_1.jpg",
        "https://bagongkia.github.io/react-image-picker/0759b6e526e3c6d72569894e58329d89.jpg",
        Image.open("./images/car_1.jpg",),
        np.array(Image.open("./images/car_1.jpg",)),
    ],
    captions=["A cat", "Another cat", "Oh look, a cat!", "Guess what, a cat..."],
    
)
=======
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
                liked_df.to_csv("./data/liked.csv", index=False)

    st.sidebar.write(f"Liked Goods {len(liked_df)}/10")
    progress_bar = st.sidebar.progress((min(len(liked_df) / 10, 1.0)))

def first():
    st.title("First Recomendation Algorithm")
    data = pd.read_csv("./data/liked.csv")
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

    # # Store the initial value of widgets in session state
    # if "visibility" not in st.session_state:
    #     st.session_state.visibility = "visible"
    #     st.session_state.disabled = False

    # col1, col2 = st.columns(2)

    # with col1:
    #     st.checkbox("Disable selectbox widget", key="disabled")
    #     st.radio(
    #         "Set selectbox label visibility 👉",
    #         key="visibility",
    #         options=["visible", "hidden", "collapsed"],
    #     )

    # with col2:
    #     option = st.selectbox(
    #         "How would you like to be contacted?",
    #         ("Email", "Home phone", "Mobile phone"),
    #         label_visibility=st.session_state.visibility,
    #         disabled=st.session_state.disabled,
    #     )
>>>>>>> dd7cdf0 (Initial)
