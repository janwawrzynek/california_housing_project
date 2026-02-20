import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
housing_full = pd.read_parquet("datasets/housing/housing_full.parquet")
housing_filtered = pd.read_parquet("datasets/housing/housing_filtered.parquet")

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
plt.title('Full dataset with capped values')
plt.show()
 
# We can see that there are some median incomes capped at [500000, 450000, 350000, 280000] in the dataset, this is an artifact of the data collection process.
# To resolve this in mai, I have filtered out these capped values from the dataset, to prevent the model from learning this "artificial" cap and thus improving generalization to other datasets.
# Lets see if this solves are problem by plotting the same graph again without the capped values.
housing_filtered.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
plt.title('Filtered dataset without capped values')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('full vs filtered dataset')
ax1.scatter(housing_full["median_income"], housing_full["median_house_value"], alpha=0.1)
ax2.scatter(housing_filtered["median_income"], housing_filtered["median_house_value"], alpha=0.1)
plt.tight_layout() # Prevents overlapping
plt.show()

print(f"Full rows: {len(housing_full)}")
print(f"Filtered rows: {len(housing_filtered)}")