import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

# 1. Load model and sample data
model_linear = joblib.load("models/housing_model_linear.pkl")
model_xgb = joblib.load("models/housing_model_xgboost.pkl")



