import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
housing_full = pd.read_parquet("datasets/housing/housing_full.parquet")

housing_full.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing_full["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.show()
 

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing_full[attributes], figsize=(12, 8))
 
housing_full.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
plt.show()
 
# We can see that the 