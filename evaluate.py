import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.pipeline import create_full_pipeline

# 1. Load Data (Ensuring we use the combined block)
X_train_full = pd.read_csv("datasets/housing/housing_train_full_features.csv")
y_train_full = pd.read_csv("datasets/housing/housing_train_full_labels.csv").iloc[:, 0]

# 2. Stratification Logic
income_cat = pd.cut(X_train_full["median_income"],
                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                    labels=[1, 2, 3, 4, 5])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Tournament Loop
model_types = ["linear", "xgboost", "random_forest", "decision_tree"] 
plot_data = []

print("--- Starting Model Tournament (5-Fold Stratified CV) ---")

for m_type in model_types:
    print(f"\nEvaluating {m_type.upper()}...")
    pipeline_blueprint = create_full_pipeline(model_type=m_type)
    
    start_time = time.perf_counter()
    cv_scores = cross_val_score(
        pipeline_blueprint, 
        X_train_full, 
        y_train_full,
        scoring="neg_root_mean_squared_error",
        cv=skf.split(X_train_full, income_cat)
    )
    duration = time.perf_counter() - start_time
    rmse_scores = -cv_scores

    # --- CHANGE 1: Capture Std Dev in plot_data ---
    plot_data.append({
        "Model": m_type.upper(),
        "RMSE": rmse_scores.mean(),
        "Std": rmse_scores.std(),   # Added this
        "Time": duration 
    })
    
    print(f"  Mean RMSE: ${rmse_scores.mean():,.2f}")
    print(f"  Std Dev:   ${rmse_scores.std():,.2f}")
    print(f"  Training Time: {duration:.2f} seconds")

# 4. Create Plot
df_results = pd.DataFrame(plot_data)
fig, ax1 = plt.subplots(figsize=(10, 6))

# --- CHANGE 2: Add yerr=df_results["Std"] ---
color_rmse = 'tab:blue'
ax1.set_xlabel('Model Type', fontweight='bold')
ax1.set_ylabel('Mean RMSE (Lower is Better)', color=color_rmse, fontweight='bold')

ax1.bar(
    df_results["Model"], 
    df_results["RMSE"], 
    yerr=df_results["Std"],       # This draws the error bars
    capsize=10,                   # Adds the horizontal 'caps'
    color=color_rmse, 
    alpha=0.6, 
    width=0.4,
    label='Mean RMSE ± Std Dev'
)
ax1.tick_params(axis='y', labelcolor=color_rmse)

# Plot Time (Line Chart remains same)
ax2 = ax1.twinx()
color_time = 'tab:red'
ax2.set_ylabel('Training Time (Seconds)', color=color_time, fontweight='bold')
ax2.plot(df_results["Model"], df_results["Time"], color=color_time, marker='o', linewidth=2, markersize=8, label='Train Time')
ax2.tick_params(axis='y', labelcolor=color_time)

plt.title('Performance, Stability & Efficiency Benchmark', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.3)
fig.tight_layout()

plt.savefig("notebooks/model_benchmarking.png")
print("\n✓ Benchmarking plot with error bars saved.")