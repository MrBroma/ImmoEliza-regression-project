# import of the necessary libraries
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# function to load the scraping data in json format
def load_file():
    data = pd.read_json("data/final_dataset.json")
    return data




    



if __name__ == "__main__":
    print(load_file())