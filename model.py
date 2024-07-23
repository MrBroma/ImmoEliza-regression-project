from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error

def prepare_data(data_sales):
    columns_to_keep = [
        'BathroomCount', 'BedroomCount', 'LivingArea', 'SurfaceOfPlot', 'ToiletCount', 'State_Encoded',
        'District', 'FloodingZone_Encoded', 'Kitchen', 'Province', 'Region', 'SubtypeOfProperty', 'PEB_Encoded', 
        'Garden', 'NumberOfFacades', 'SwimmingPool', 'Price', 'Locality'
    ]
    
    data_sales = data_sales[columns_to_keep]

    ode_cols = ['BathroomCount', 'BedroomCount', 'LivingArea', 'SurfaceOfPlot', 'ToiletCount', 'State_Encoded', 'FloodingZone_Encoded']
    ohe_cols = ['District', 'Kitchen', 'Province', 'Region', 'SubtypeOfProperty', 'Locality']
    num_cols = ['Garden', 'NumberOfFacades', 'SwimmingPool', 'PEB_Encoded'] 

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

    return col_trans

def train_model(data_sales, col_trans):
    X = data_sales.drop('Price', axis=1)
    y = data_sales['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

    pipeline = Pipeline(steps=[
        ('preprocessing', col_trans),
        ('model', GradientBoostingRegressor())
    ])

    param_grid = {
        'preprocessing__num_p__impute__n_neighbors': [3, 5, 7, 9, 11]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_k = grid_search.best_params_['preprocessing__num_p__impute__n_neighbors']
    
    pipeline.set_params(preprocessing__num_p__impute__n_neighbors=best_k)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)

    return train_score, test_score, mae
