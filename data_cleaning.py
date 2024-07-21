import pandas as pd
import numpy as np

data = pd.read_json("final_dataset.json")

data.drop_duplicates("PropertyId",inplace=True)
data.update(data[["BathroomCount","Fireplace","Furnished","Garden","GardenArea","SwimmingPool","Terrace","ToiletCount"]].fillna(0))

data.drop(data[data.BathroomCount > data.BedroomCount].index,inplace=True)
data.drop(data[data.ConstructionYear > 2033].index,inplace=True)
data.drop(data[data.GardenArea > data.SurfaceOfPlot].index,inplace=True)
data.drop(data[data.PostalCode < 1000].index,inplace=True)
data.drop(data[data.NumberOfFacades > 4].index,inplace=True)
data.drop(data[data.Price > 15000000].index,inplace=True)
data.drop(data[data.ToiletCount > 58].index,inplace=True)
data.drop(data[data.ShowerCount > 58].index,inplace=True)
data.drop(data[data.LivingArea > 8800].index, inplace=True)
# data.drop(data[data.TypeOfSale == "annuity_monthly_amount"].index,inplace=True)
# data.drop(data[data.TypeOfSale == "annuity_without_lump_sum"].index,inplace=True)
# data.drop(data[data.TypeOfSale == "annuity_lump_sum"].index,inplace=True)

sale_data = data[data.TypeOfSale == "residential_sale"]
rent_data = data[data.TypeOfSale == "residential_monthly_rent"]

BathroomCount
# Fireplace
# Furnished
# Garden
GardenArea
# SwimmingPool
# Terrace
ToiletCount
