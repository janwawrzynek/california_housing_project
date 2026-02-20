import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

# 1. Load model and sample data
model = joblib.load("models/housing_model.pkl")
df = pd.read_csv("datasets/housing/housing.csv")
X = df.drop("median_house_value", axis=1)
X_sample = X.sample(500, random_state=42) # Increased sample for better stats

# 2. Get names and transform data
preprocessing = model.named_steps["preprocessing"]
X_transformed = preprocessing.transform(X_sample)
feature_names = preprocessing.get_feature_names_out()

# 3. Calculate SHAP values
explainer = shap.TreeExplainer(model.named_steps["regressor"])
shap_values_obj = explainer(X_transformed) # Get the full Explanation object
shap_values = shap_values_obj.values

# 4. AGGREGATION LOGIC
# Identify which columns belong to the Geo clusters
geo_indices = [i for i, name in enumerate(feature_names) if "geo__Cluster" in name]
other_indices = [i for i, name in enumerate(feature_names) if "geo__Cluster" not in name]

# Sum SHAP values for the geo columns
combined_values = np.column_stack([
    shap_values[:, other_indices],
    shap_values[:, geo_indices].sum(axis=1)
])

# Sum the actual data values (optional, for visualization)
combined_data = np.column_stack([
    X_transformed[:, other_indices],
    np.zeros(len(X_transformed)) # Placeholder for the "combined" feature data
])

combined_names = [feature_names[i] for i in other_indices] + ["Total Geographic Impact"]

# 5. PROFESSIONAL PLOTTING
plt.figure(figsize=(10, 6))
# We create a new Explanation object for the aggregated data to use the better '.bar' plot
clean_explanation = shap.Explanation(
    values=combined_values,
    data=combined_data,
    feature_names=combined_names
)

print("Saving clean horizontal feature importance plot...")
shap.plots.bar(clean_explanation, max_display=15, show=False)
plt.tight_layout()
plt.savefig("notebooks/feature_importance_clean.png")
plt.show()