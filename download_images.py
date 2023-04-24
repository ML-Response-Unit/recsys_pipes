import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

data = pd.read_csv("./data/cars_about.csv").dropna()
# List of car names
car_names = [f"{data.iloc[index].car_model} {data.iloc[index].exteriorColor}".replace("/", " ") for index in range(len(data))]


# Loop through each car name and download the first image in the search results
for car_name in car_names:
    # Download the image to a file
    filename = os.path.join(f"./images/{car_name}.jpg")
    if os.path.exists(filename): continue
    
    # URL of the Bing Image Search with the car name as the search query
    url = f"https://www.bing.com/images/search?q={car_name}&form=HDRSC2"

    # Send an HTTP GET request to the search URL
    response = requests.get(url)

    # Check if the response was successful
    if response.status_code == 200:
        # Parse the response HTML using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the first image result and extract its URL
        image_div = soup.find("div", {"class": "img_cont"})
        if image_div is not None:
            image_url = image_div.find("img")["src"]

            with open(filename, "wb") as f:
                response = requests.get(image_url)
                f.write(response.content)
                print(f"{filename} downloaded successfully!")
        else:
            print(f"No image found for {car_name}")
    else:
        print(f"Error: {response.status_code}")