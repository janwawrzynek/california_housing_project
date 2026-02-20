# main.py
import joblib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tarfile
import urllib.request
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import pandas as pd
from src.pipeline import create_full_pipeline
from sklearn.model_selection import train_test_split


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


if __name__ == "__main__":
    print("Loading data...")
    housing_full = load_housing_data()
    housing_full.to_parquet("datasets/housing/housing_full.parquet")
    # After noticing in the exploration.py file that there are some median incomes that are capped, I will remove these from the dataset, to prevent the model
    # from learning this "artificial" cap and thus improving generalization to other datasets.
    cap_values = [500000, 450000, 350000, 280000] 


    # Create income categories for stratification (predefined bins to avoid data snooping)
    housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
    
    housing_filtered = housing_full[~housing_full["median_house_value"].isin(cap_values)].reset_index(drop=True)

    

    
    # 1. First split: 80% train+val, 20% test (stratified)
    strat_train_val_set, strat_test_set = train_test_split(
    housing_filtered, test_size=0.2, stratify=housing_filtered["income_cat"],
    random_state=42)

    
    # 2. Second split: 75% train, 25% val (stratified on train+val only)
    strat_train_set, strat_val_set = train_test_split(
    strat_train_val_set, test_size=0.25, stratify=strat_train_val_set["income_cat"],
    random_state=42)
    # Won't use the income_cat feature for modeling, so we drop it now to avoid data leakage
    for set_ in (strat_train_set, strat_val_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Make a copy of the originial data so that we can use it later
    # Separate the predictor features and the target labels as we don't want to include the target in the feature engineering steps
    housing_train_features = strat_train_set.drop("median_house_value", axis=1) # drop() creates a copy of the default and does not affect the origninal.
    housing_train_labels = strat_train_set["median_house_value"].copy()

    # Save the Datasets to prevent data leakage and allow for reproducability.
    housing_train_features.to_csv("datasets/housing/housing_train_features.csv", index=False)
    housing_train_labels.to_csv("datasets/housing/housing_train_labels.csv", index=False)
    # Test
    housing_test_features = strat_test_set.drop("median_house_value", axis=1)
    housing_test_labels = strat_test_set["median_house_value"].copy()

    housing_test_features.to_csv("datasets/housing/housing_test.csv", index=False)
    housing_test_labels.to_csv("datasets/housing/housing_test_labels.csv", index=False)
    print("✓ Data splits saved to datasets/housing/ using housing_* convention")
    
    
    print("Initializing high-performance XGBoost pipeline...")
    model = create_full_pipeline()
    
    print("Training...")
    model.fit(housing_train_features, housing_train_labels)
    
    print("Saving model to models/housing_model.pkl...")
    joblib.dump(model, "models/housing_model.pkl")
    print("Success!")