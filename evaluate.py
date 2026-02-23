import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error


# 1. Load models 
model_linear = joblib.load("models/housing_model_linear.pkl")
model_xgb = joblib.load("models/housing_model_xgboost.pkl")
model_types = {"linear": model_linear, "xgboost": model_xgb}

# 2. Load sample data
housing_val_features = pd.read_csv("datasets/housing/housing_val_features.csv")
housing_val_labels = pd.read_csv("datasets/housing/housing_val_labels.csv")

# 3. Get predictions
for model_name, model in model_types.items():
    predictions = model.predict(housing_val_features)
    rmse = root_mean_squared_error(housing_val_labels, predictions)
    print(f"{model_name} RMSE: {rmse:.2f}")