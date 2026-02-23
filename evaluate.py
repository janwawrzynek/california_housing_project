import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.pipeline import create_full_pipeline # Using the blueprint factory

# 1. Load the FULL training data block (80% of total data)
# This is the combined set you saved in main.py
X_train_full = pd.read_csv("datasets/housing/housing_train_full_features.csv")
y_train_full = pd.read_csv("datasets/housing/housing_train_full_labels.csv").iloc[:, 0]

# 2. Recreate income categories for StratifiedKFold logic
# We must stratify by the same bins used in the original split
income_cat = pd.cut(X_train_full["median_income"],
                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                    labels=[1, 2, 3, 4, 5])

# 3. Setup the StratifiedKFold iterator
# 5 folds is the industry standard for medium-sized datasets
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Run the Model Tournament
model_types = ["linear", "xgboost", "random_forest", "decision_tree"] # You can add more model types here as you experiment
results = {}

print("--- Starting Model Tournament (5-Fold Stratified CV) ---")

for m_type in model_types:
    print(f"\nEvaluating {m_type.upper()}...")
    
    # Get a fresh, UNTRAINED blueprint
    pipeline_blueprint = create_full_pipeline(model_type=m_type)
    
    # Run Cross-Validation
    # This fits/evaluates the blueprint 5 times on 5 different folds
    cv_scores = cross_val_score(
        pipeline_blueprint, 
        X_train_full, 
        y_train_full,
        scoring="neg_root_mean_squared_error",
        cv=skf.split(X_train_full, income_cat)
    )
    
    # Convert negative scores to positive RMSE
    rmse_scores = -cv_scores
    results[m_type] = rmse_scores
    
    print(f"  Mean RMSE: ${rmse_scores.mean():,.2f}")
    print(f"  Std Dev:   ${rmse_scores.std():,.2f}")

    # ... [Data Cleaning & Splitting Logic Above] ...

    # --- PRODUCTION TOGGLE ---
    # Set this to True only AFTER you have confirmed the winner in evaluate.py
    RUN_FINAL_FIT = False

    if RUN_FINAL_FIT:
        print("\n--- Training Final Production Model ---")
        final_model = create_full_pipeline(model_type="xgboost")
        final_model.fit(X_train_full, y_train_full)
        joblib.dump(final_model, "models/housing_model.pkl")
        print("✓ Production model updated.")
    else:
        print("\n[INFO] Skipping final fit. Run evaluate.py next to confirm model choice.")
# 5. Final Comparison
#print("\n" + "="*30)
#print("FINAL RECOMMENDATION")
##print("="*30)
#best_model = min(results, key=lambda k: results[k].mean())
#improvement = (results["linear"].mean() - results["xgboost"].mean()) / results["linear"].mean() * 100

#print(f"Winner: {best_model.upper()}")
#print(f"Performance Gain over Baseline: {improvement:.2f}%")