import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from xgboost import XGBRegressor  
from sklearn.linear_model import LinearRegression

# Import custom transformers and helper functions
from src.transformers import ClusterSimilarity, column_ratio, ratio_name

def get_preprocessing_pipeline():
    """Builds the full ColumnTransformer for the housing data."""
    
    # 1. Setup specialized pipelines
    # Logic is isolated so changes here propagate to both model types automatically
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

    # 10 clusters for geographic RBF similarity (The "Physics" touch)
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )

    # 2. Combine into ColumnTransformer
    # This acts as the 'strict logical safeguard' for feature engineering
    preprocessing = ColumnTransformer([
        ("bedrooms_ratio", ratio_pipeline, ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline, ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline, ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ], remainder=StandardScaler()) 

    return preprocessing

def create_full_pipeline(model_type="xgboost", hyperparams=None):
    """
    Creates an end-to-end pipeline with a switchable regressor.
    Returns an UNTRAINED blueprint ready for StratifiedKFold cross-validation.
    """
    preprocessing = get_preprocessing_pipeline()
    
    if model_type == "xgboost":
        # Using a dictionary for defaults ensures scientific reproducibility
        if hyperparams is None:
            hyperparams = {
                'n_estimators': 500, 
                'learning_rate': 0.05, 
                'max_depth': 6, 
                'n_jobs': -1,
                'random_state': 42 # Added for consistent CV results
            }
        regressor = XGBRegressor(**hyperparams)
    
    elif model_type == "linear":
        regressor = LinearRegression()
    
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        if hyperparams is None:
            hyperparams = {
                'n_estimators': 100, 
                'max_depth': None, 
                'n_jobs': -1,
                'random_state': 42 # Added for consistent CV results
            }
        regressor = RandomForestRegressor(**hyperparams)

    elif model_type == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor
        if hyperparams is None:
            hyperparams = {
                'max_depth': None, 
                'random_state': 42 # Added for consistent CV results
            }
        regressor = DecisionTreeRegressor(**hyperparams)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected 'xgboost' or 'linear'.")

    # Coupling preprocessing and regressor prevents data leakage during cross_val_score
    return Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", regressor)
    ])