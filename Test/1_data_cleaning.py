#######################CLEANING########################3

# load json as a dataframe with pandas
data = pd.read_json("data/final_dataset.json")

postal_codes_to_keep = pd.read_csv('data/postalcode_be.csv')
postal_codes_to_keep_list = postal_codes_to_keep['Code postal'].tolist()
data = data[data['PostalCode'].isin(postal_codes_to_keep_list)]

data.drop_duplicates("PropertyId",inplace=True)
data.drop(data[data.ConstructionYear > 2033].index,inplace=True)
data.drop(data[data.PostalCode < 1000].index,inplace=True)
data.drop(data[data.GardenArea > 200000].index,inplace=True)
data.drop(data[data.LivingArea > 4500].index, inplace=True)
data.drop(data[data.Price > 4000000].index,inplace=True)
data.drop(data[data.NumberOfFacades > 8].index,inplace=True)
data.drop(data[data.ShowerCount > 58].index,inplace=True)
data.drop(data[data.SurfaceOfPlot > 200000].index,inplace=True)
data.drop(data[data.ToiletCount > 25].index,inplace=True)
data.drop(data[data.District == None].index,inplace=True)
data.drop(data[data.Province == None].index,inplace=True)
data.drop(data[data.Region == None].index,inplace=True)
data.drop(data[data.Locality == None].index,inplace=True)
data.drop(data[data.TypeOfSale == "annuity_monthly_amount"].index,inplace=True)
data.drop(data[data.TypeOfSale == "annuity_without_lump_sum"].index,inplace=True)
data.drop(data[data.TypeOfSale == "annuity_lump_sum"].index,inplace=True)
data.drop(data[data.TypeOfSale == "homes_to_build"].index,inplace=True)

data_sales = data[data.TypeOfSale == "residential_sale"]
data_rent = data[data.TypeOfSale == "residential_monthly_rent"]

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
# order of the PEB
peb_order = ['A++', 'A+', 'B', 'C', 'D', 'E', 'F', 'G', 'Unknown']
ordinal_encoder = OrdinalEncoder(categories=[peb_order])
data_sales.loc[:, 'PEB_Encoded'] = ordinal_encoder.fit_transform(data_sales[['PEB']])

state_order = ['AS_NEW', 'JUST_RENOVATED', 'GOOD', 'TO_BE_DONE_UP', 'TO_RENOVATE', 'TO_RESTORE', 'Unknown']
ordinal_encoder = OrdinalEncoder(categories=[state_order])
data_sales.loc[:, 'State_Encoded'] = ordinal_encoder.fit_transform(data_sales[['StateOfBuilding']])

def transform_flooding_zone(zone):
    if zone == 'NON_FLOOD_ZONE':
        return 0
    else:
        return 1

data_sales['FloodingZone_Encoded'] = data_sales['FloodingZone'].apply(transform_flooding_zone)
data_sales['LivingArea'].fillna(data_sales['LivingArea'].median(), inplace=True)
data_sales = data_sales.drop(columns=['Url', 'Country', 'TypeOfSale', 'PropertyId', 'TypeOfProperty', 'PostalCode'])



#############################MODEL ML##########################
columns_to_keep = [
    'BathroomCount', 'BedroomCount', 'LivingArea', 'SurfaceOfPlot', 'ToiletCount', 'PEB', 'State_Encoded',
    'District', 'FloodingZone', 'Kitchen', 'Province', 'Region', 'SubtypeOfProperty', 'Garden', 'NumberOfFacades',
    'SwimmingPool', 'Price', 'Locality'
]

data_sales = data_sales[columns_to_keep]

ode_cols = ['BathroomCount', 'BedroomCount', 'LivingArea', 'SurfaceOfPlot', 'ToiletCount', 'State_Encoded']
ohe_cols = ['District', 'FloodingZone', 'Kitchen', 'Province', 'Region', 'SubtypeOfProperty', 'PEB', 'Locality']
num_cols = ['Garden', 'NumberOfFacades', 'SwimmingPool'] 

num_pipeline = Pipeline(steps=[
    ('impute', KNNImputer()),
    ('scaler', StandardScaler())
])
ode_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler()) 
])
ohe_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

col_trans = ColumnTransformer(transformers=[
    ('num_p', num_pipeline, num_cols),
    ('ode_p', ode_pipeline, ode_cols),
    ('ohe_p', ohe_pipeline, ohe_cols)
], remainder='passthrough', n_jobs=-1)

# Preprocessing pipeline
pipeline = Pipeline(steps=[
    ('preprocessing', col_trans),
    ('model', GradientBoostingRegressor())
])

X = data_sales.drop('Price', axis=1)
y = data_sales['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

param_grid = {
    'preprocessing__num_p__impute__n_neighbors': [3, 5, 7, 9, 11]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_k = grid_search.best_params_['preprocessing__num_p__impute__n_neighbors']
# print("Best k for KNNImputer:", best_k)

pipeline.set_params(preprocessing__num_p__impute__n_neighbors=best_k)
pipeline.fit(X_train, y_train)
y_pred_lr = pipeline.predict(X_test)



# Calcul des scores
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred_lr)

print("Train Score: ", train_score)
print("Test Score: ", test_score)
print("Mean Absolute Error: ", mae)














