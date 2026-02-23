import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from xgboost import XGBRegressor  
# Compare results with the LinearRegressor baseline
from sklearn.linear_model import LinearRegression


# Import  custom transformers and helper functions
from src.transformers import ClusterSimilarity, column_ratio, ratio_name

def get_preprocessing_pipeline():
    """Builds the full ColumnTransformer for the housing data."""
    
    # 1. Setup specialized pipelines
    ratio_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )

    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler()
    )

    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )

    # 2. Combine into ColumnTransformer
    preprocessing = ColumnTransformer([
        ("bedrooms_ratio", ratio_pipeline, ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline, ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline, ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ], remainder=StandardScaler()) # Remainder is housing_median_age

    return preprocessing

def create_full_pipeline(hyperparams=None):
    """Creates the final end-to-end model pipeline."""
    if hyperparams is None:
        # High-performance defaults
        hyperparams = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_jobs': -1 # Utilize all CPU cores
        }
    
    preprocessing = get_preprocessing_pipeline()
    
    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", XGBRegressor(**hyperparams))
    ])
    
    return full_pipeline