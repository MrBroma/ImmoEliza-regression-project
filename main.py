import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedKFold
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor  # Utilisez KNeighborsClassifier pour la classification

import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

from data_cleaning import load_data, clean_data, split_data, preprocess_sales_data
from model import prepare_data, train_model

def main():
    # Load and clean data
    data = load_data()
    data = clean_data(data)
    
    # Split data
    data_sales, data_rent = split_data(data)
    
    # Preprocess sales data
    data_sales = preprocess_sales_data(data_sales)
    
    # Prepare data for model
    col_trans = prepare_data(data_sales)
    
    # Train model
    train_score, test_score, mae = train_model(data_sales, col_trans)
    
    # Print results
    print("Train Score: ", train_score)
    print("Test Score: ", test_score)
    print("Mean Absolute Error: ", mae)

if __name__ == "__main__":
    main()
