import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from src.pipeline import create_full_pipeline
import matplotlib.pyplot as plt

# 1. Load Data (Raw features)
X_train = pd.read_csv("datasets/housing/housing_train_full_features.csv")
y_train = pd.read_csv("datasets/housing/housing_train_full_labels.csv").iloc[:, 0]
X_test = pd.read_csv("datasets/housing/housing_test.csv")
y_test = pd.read_csv("datasets/housing/housing_test_labels.csv").iloc[:, 0]

# 2. Create a fresh pipeline blueprint for tuning
# We use the full raw data and let the pipeline handle transformations
print("--- Initializing Pipeline for Tuning ---")
tuning_pipeline = create_full_pipeline(model_type="xgboost")

# 3. Define the Hyperparameter Search Space
# We target the components inside the pipeline using double underscores
param_distribs = {
    'preprocessing__geo__n_clusters': randint(low=10, high=50),
    'regressor__n_estimators': randint(low=100, high=1000),
    'regressor__learning_rate': uniform(0.01, 0.3),
    'regressor__max_depth': randint(low=3, high=10),
    'regressor__subsample': uniform(0.5, 0.5) # Range 0.5 to 1.0
}

# 4. Setup Randomized Search
rnd_search = RandomizedSearchCV(
    tuning_pipeline,
    param_distributions=param_distribs,
    n_iter=15, 
    cv=3,
    scoring="neg_root_mean_squared_error",
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("--- Starting Hyperparameter Search (This may take a few minutes) ---")
rnd_search.fit(X_train, y_train)

# 5. Get the best model
best_tuned_model = rnd_search.best_estimator_
print(f"\nBest CV RMSE: ${-rnd_search.best_score_:,.2f}")
print(f" Standard Deviation of CV Scores: ${rnd_search.cv_results_['std_test_score'][rnd_search.best_index_]:,.2f}")
print("Best Params:", rnd_search.best_params_)
joblib.dump(best_tuned_model, "models/housing_model_tuned.pkl")
print(" Tuned model saved as housing_model_tuned.pkl")

# ... (Previous code: Search, Fit, and Test Evaluation) ...

# 1. DEFINE YOUR TOURNAMENT BENCHMARK (Manually enter from your evaluate.py results)
# Replace these numbers with the ones from your 'Evaluating XGBOOST' terminal output
benchmark_mean_rmse = 38912.20  # Example value
benchmark_std_rmse = 1267.88  # Example value

# 2. EXTRACT TUNED RESULTS
tuned_mean_rmse = -rnd_search.best_score_
best_index = rnd_search.best_index_
tuned_std_rmse = rnd_search.cv_results_['std_test_score'][best_index]


labels = ['Baseline Model', 'Final Tuned Model']
means = [benchmark_mean_rmse, tuned_mean_rmse]
errors = [benchmark_std_rmse, tuned_std_rmse]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, means, yerr=errors, capsize=12, 
              color=['#6c757d', '#28a745'], alpha=0.8) # Grey for old, Green for new

# Styling
ax.set_ylabel('Mean RMSE (Lower is Better)', fontweight='bold')
ax.set_title('Impact of Fine-Tuning on XGBoost Performance', fontsize=14, fontweight='bold')
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Add exact values on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2000,
            f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("notebooks/tuning_success_comparison.png")
print("\n✓ Comparison plot saved to notebooks/tuning_success_comparison.png")