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
