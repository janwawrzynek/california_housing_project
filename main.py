import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import tarfile
import urllib.request
from sklearn.model_selection import train_test_split

# Import your factory and custom logic
from src.pipeline import create_full_pipeline

def load_housing_data():
    """Fetches and loads the raw housing data."""
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

if __name__ == "__main__":
    # --- 1. DATA INGESTION ---
    print("Loading raw data...")
    housing_full = load_housing_data()

    # --- 2. FEATURE ENGINEERING & CLEANING ---
    # Create income categories for stratification to avoid data snooping
    housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
    
    # Remove artificial caps identified in exploration.py to improve generalization
    cap_values = [450000, 350000, 280000] 
    max_income = 500000
    housing_filtered = housing_full[housing_full["median_house_value"] < max_income].copy()
    housing_filtered = housing_filtered[~housing_filtered["median_house_value"].isin(cap_values)].reset_index(drop=True)

    # --- 3. DATA SPLITTING (THE SAFEGUARD) ---
    # First split: 80% Training Block (for CV and final fit), 20% Test Set (Locked away)
    strat_train_full_set, strat_test_set = train_test_split(
        housing_filtered, test_size=0.2, stratify=housing_filtered["income_cat"],
        random_state=42)

    # Remove the stratification helper column
    for set_ in (strat_train_full_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Prepare features and labels
    X_train_full = strat_train_full_set.drop("median_house_value", axis=1)
    y_train_full = strat_train_full_set["median_house_value"].copy()
    
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    # --- 4. PERSISTENCE ---
    # Save the split data so evaluate.py and audit.py can use them
    Path("datasets/housing").mkdir(parents=True, exist_ok=True)
    X_train_full.to_csv("datasets/housing/housing_train_full_features.csv", index=False)
    y_train_full.to_csv("datasets/housing/housing_train_full_labels.csv", index=False)
    X_test.to_csv("datasets/housing/housing_test.csv", index=False)
    y_test.to_csv("datasets/housing/housing_test_labels.csv", index=False)
    print("✓ Production data splits saved.")

    # --- 5. FINAL PRODUCTION FIT ---
    # We now fit on 100% of the non-test data for maximum predictive power.
    print("\nTraining final production model (XGBoost)...")
    # ... [Data Cleaning & Splitting Logic Above] ...

    # --- PRODUCTION TOGGLE ---
    # Set this to True only AFTER you have confirmed the winner in evaluate.py
    RUN_FINAL_FIT = True

    if RUN_FINAL_FIT:
        print("\n--- Training Final Production Model ---")
        final_model = create_full_pipeline(model_type="xgboost")
        final_model.fit(X_train_full, y_train_full)
        joblib.dump(final_model, "models/housing_model.pkl")
        print("✓ Production model updated.")
    else:
        print("\n[INFO] Skipping final fit. Run evaluate.py next to confirm model choice.")