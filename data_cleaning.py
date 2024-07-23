import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def load_data():
    data = pd.read_json("data/final_dataset.json")
    postal_codes_to_keep = pd.read_csv('data/postalcode_be.csv')
    postal_codes_to_keep_list = postal_codes_to_keep['Code postal'].tolist()
    data = data[data['PostalCode'].isin(postal_codes_to_keep_list)]
    return data

def clean_data(data):
    data.drop_duplicates("PropertyId", inplace=True)
    data = data[data.ConstructionYear <= 2033]
    data = data[data.PostalCode >= 1000]
    data = data[data.GardenArea <= 200000]
    data = data[data.LivingArea <= 4500]
    data = data[data.Price <= 4000000]
    data = data[data.NumberOfFacades <= 8]
    data = data[data.ShowerCount <= 58]
    data = data[data.SurfaceOfPlot <= 200000]
    data = data[data.ToiletCount <= 25]
    data = data.dropna(subset=['District', 'Province', 'Region', 'Locality'])
    data = data[~data.TypeOfSale.isin(["annuity_monthly_amount", "annuity_without_lump_sum", "annuity_lump_sum", "homes_to_build"])]
    return data

def split_data(data):
    data_sales = data[data.TypeOfSale == "residential_sale"]
    data_rent = data[data.TypeOfSale == "residential_monthly_rent"]
    return data_sales, data_rent

def preprocess_sales_data(data_sales):
    data_sales.drop(['MonthlyCharges'], axis=1, inplace=True)
    data_sales['Fireplace'].fillna(0, inplace=True)
    data_sales['Garden'].fillna(0, inplace=True)
    data_sales['GardenArea'].fillna(data_sales['GardenArea'].median(), inplace=True)
    data_sales['Furnished'].fillna(0, inplace=True)
    data_sales['SwimmingPool'].fillna(0, inplace=True)
    data_sales['ShowerCount'].fillna(0, inplace=True)
    data_sales['FloodingZone'].fillna('NON_FLOOD_ZONE', inplace=True)
    data_sales['SurfaceOfPlot'].fillna(data_sales['SurfaceOfPlot'].median(), inplace=True)
    data_sales['Kitchen'].fillna('NOT_INSTALLED', inplace=True)
    data_sales['Terrace'].fillna(0, inplace=True)
    data_sales['NumberOfFacades'].fillna(2, inplace=True)
    data_sales['ToiletCount'].fillna(0, inplace=True)
    data_sales['BathroomCount'].fillna(data_sales['BathroomCount'].median(), inplace=True)
    data_sales['StateOfBuilding'].fillna('GOOD', inplace=True)
    data_sales['PEB'].fillna('Unknown', inplace=True)

    keep_PEB = ['A++', 'A+', 'B', 'C', 'D', 'E', 'F', 'G']
    data_sales = data_sales[data_sales['PEB'].isin(keep_PEB)]
    
    # Encodage ordinal pour PEB
    peb_order = ['A++', 'A+', 'B', 'C', 'D', 'E', 'F', 'G', 'Unknown']
    ordinal_encoder_peb = OrdinalEncoder(categories=[peb_order])
    data_sales['PEB_Encoded'] = ordinal_encoder_peb.fit_transform(data_sales[['PEB']])

    # Encodage ordinal pour l'état du bâtiment
    state_order = ['AS_NEW', 'JUST_RENOVATED', 'GOOD', 'TO_BE_DONE_UP', 'TO_RENOVATE', 'TO_RESTORE', 'Unknown']
    ordinal_encoder_state = OrdinalEncoder(categories=[state_order])
    data_sales['State_Encoded'] = ordinal_encoder_state.fit_transform(data_sales[['StateOfBuilding']])

    data_sales['FloodingZone_Encoded'] = data_sales['FloodingZone'].apply(lambda zone: 0 if zone == 'NON_FLOOD_ZONE' else 1)
    data_sales['LivingArea'].fillna(data_sales['LivingArea'].median(), inplace=True)
    data_sales = data_sales.drop(columns=['Url', 'Country', 'TypeOfSale', 'PropertyId', 'TypeOfProperty', 'PostalCode'])

    return data_sales
