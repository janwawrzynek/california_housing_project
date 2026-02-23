import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from scipy import stats

# 1. Load the finalized model and the "locked" test data
print("Loading production model and test set...")
model = joblib.load("models/housing_model.pkl")
X_test = pd.read_csv("datasets/housing/housing_test.csv")
y_test = pd.read_csv("datasets/housing/housing_test_labels.csv").iloc[:, 0]

# 2. Generate final predictions
predictions = model.predict(X_test)

# 3. Calculate the Headline RMSE
final_rmse = root_mean_squared_error(y_test, predictions)
print(f"\n--- FINAL TEST RESULTS ---")
print(f"Final Test RMSE: ${final_rmse:,.2f}")

# 4. Calculate the 95% Confidence Interval (The Scientific Proof)
confidence = 0.95
squared_errors = (predictions - y_test) ** 2
m = len(squared_errors)

# Use the t-distribution for the interval
mean = squared_errors.mean()
t_score = stats.t.ppf((1 + confidence) / 2, df=m - 1)
margin_of_error = t_score * (squared_errors.std(ddof=1) / np.sqrt(m))

# Convert the interval of squared errors back to RMSE scale
rmse_lower = np.sqrt(mean - margin_of_error)
rmse_upper = np.sqrt(mean + margin_of_error)

print(f"95% Confidence Interval: [${rmse_lower:,.2f}, ${rmse_upper:,.2f}]")