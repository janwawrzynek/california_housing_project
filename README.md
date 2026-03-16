# California Housing Price Prediction

An end-to-end machine learning pipeline for predicting median house prices across California districts. The project covers the full data science workflow — from raw data ingestion and ETL preprocessing through model comparison, hyperparameter optimisation, and production deployment as a containerised REST API.

---

## Results

| Model | Cross-Val RMSE | Training Time |
|---|---|---|
| Linear Regression | $58,854.70 | 0.45s |
| Decision Tree | $58,562.56 | 1.32s |
| Random Forest |$41,167.60 | 9.56s |
| **XGBoost (baseline)** | **$38,912** | 6.42 |
| **XGBoost (tuned)** | **$37,480** | Fast |

XGBoost was selected as the final model based on its superior RMSE and training efficiency. Hyperparameter tuning via `RandomizedSearchCV` reduced RMSE by a further **$1,432** (~3.7%).

---

## Project Structure

```
.
├── main.py              # Data ingestion, train/test split, ETL preprocessing
├── exploration.py       # Exploratory data analysis and visualisations
├── evaluate.py          # Model comparison via cross-validation
├── fine_tuning.py       # XGBoost hyperparameter optimisation (RandomizedSearchCV)
├── explain.py           # Model explainability (feature importances)
├── server.py            # FastAPI REST API for live predictions
├── test.py              # API and pipeline tests
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── datasets/            # Raw and processed data
├── models/              # Serialised trained models
├── notebooks/           # Exploratory Jupyter notebooks
└── src/                 # Core pipeline modules
```

---

## Pipeline Overview

```
Raw Data
   │
   ▼
main.py — Ingest → Train/Test Split → ETL (clean, transform, feature engineer) → validate
   │
   ▼
exploration.py — Distribution analysis, correlation, geographic visualisation
   │
   ▼
evaluate.py — Cross-validate: Linear Regression, Decision Tree, Random Forest, XGBoost
   │
   ▼
fine_tuning.py — RandomizedSearchCV on XGBoost → serialise best model
   │
   ▼
explain.py — Feature importance analysis
   │
   ▼
server.py — FastAPI endpoint → Docker container
```

---

## Installation & Usage

### 1. Clone and install dependencies

```bash
git clone https://github.com/janwawrzynek/california_housing_project.git
cd california_housing_project
pip install -r requirements.txt
```

### 2. Fetch and preprocess the data

Downloads the dataset, performs the train/test split, and runs the full ETL pipeline including data cleaning, transformation, and feature engineering. Strict data validation is applied at each stage before the processed data is written to `datasets/`.

```bash
python main.py
```

### 3. Explore the data

Generates distribution plots, correlation matrices, and geographic visualisations of housing prices across California.

```bash
python exploration.py
```

### 4. Compare models

Trains and cross-validates Linear Regression, Decision Tree, Random Forest, and XGBoost. Prints RMSE scores and training times for each.

```bash
python evaluate.py
```

### 5. Fine-tune XGBoost

Runs `RandomizedSearchCV` over the XGBoost hyperparameter space and serialises the best model to `models/`.

```bash
python fine_tuning.py
```

### 6. Inspect feature importances

Outputs feature importance rankings for the tuned XGBoost model.

```bash
python explain.py
```

### 7. Launch the prediction API

Starts a FastAPI server locally. Once running, navigate to `http://localhost:8000/docs` for the interactive Swagger UI.

```bash
python server.py
```

**Example prediction request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.98,
    "AveBedrms": 1.02,
    "Population": 322.0,
    "AveOccup": 2.56,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

---

## Docker

The full application is containerised for cross-platform reproducibility.

**Build the image:**
```bash
docker build -t california-housing .
```

**Run the container:**
```bash
docker run -p 8000:8000 california-housing
```

The API will be available at `http://localhost:8000`. No local Python environment required.

---

## Dataset

The project uses the **California Housing dataset** derived from the 1990 US Census. Each row represents a census block group — the smallest geographical unit published by the US Census Bureau — with a typical population of 600–3,000 people.

    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str
---

## Tech Stack

| Tool | Role |
|---|---|
| `scikit-learn` | Preprocessing, model training, cross-validation |
| `XGBoost` | Final prediction model |
| `Pandas` / `NumPy` | Data manipulation and feature engineering |
| `Matplotlib` | Visualisation |
| `FastAPI` | REST API serving |
| `Docker` | Containerisation |
| `Git` | Version control |

---

## Requirements

```
alembic==1.18.4
anyio==4.12.1
appnope==0.1.4
argon2-cffi==25.1.0
argon2-cffi-bindings==25.1.0
arrow==1.4.0
asttokens==3.0.1
async-lru==2.1.0
attrs==25.4.0
babel==2.18.0
beautifulsoup4==4.14.3
bleach==6.3.0
certifi==2026.1.4
cffi==2.0.0
charset-normalizer==3.4.4
cloudpickle==3.1.2
colorlog==6.10.1
comm==0.2.3
contourpy==1.3.2
cramjam==2.11.0
curl_cffi==0.13.0
cycler==0.12.1
debugpy==1.8.20
decorator==5.2.1
defusedxml==0.7.1
executing==2.2.1
fastjsonschema==2.21.2
fastparquet==2025.12.0
filelock==3.20.3
fonttools==4.59.0
fqdn==1.5.1
frozendict==2.4.7
fsspec==2026.1.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.11
ipykernel==7.2.0
ipython==9.10.0
ipython_pygments_lexers==1.1.1
isoduration==20.11.0
jedi==0.19.2
Jinja2==3.1.6
joblib==1.5.3
json5==0.13.0
jsonpointer==3.0.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-events==0.12.0
jupyter-lsp==2.3.0
jupyter_client==8.8.0
jupyter_core==5.9.1
jupyter_server==2.17.0
jupyter_server_terminals==0.5.4
jupyterlab==4.5.3
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.0
kiwisolver==1.4.8
lark==1.3.1
llvmlite==0.46.0
Mako==1.3.10
MarkupSafe==3.0.3
matplotlib==3.10.3
matplotlib-inline==0.2.1
mistune==3.2.0
mpmath==1.3.0
multitasking==0.0.12
nbclient==0.10.4
nbconvert==7.17.0
nbformat==5.10.4
nest-asyncio==1.6.0
networkx==3.6.1
notebook_shim==0.2.4
numba==0.64.0
numpy==2.3.1
optuna==4.7.0
packaging==25.0
pandas==2.3.1
pandocfilters==1.5.1
parso==0.8.5
peewee==3.19.0
pexpect==4.9.0
pillow==11.3.0
platformdirs==4.5.1
prometheus_client==0.24.1
prompt_toolkit==3.0.52
protobuf==6.33.4
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.3
pyarrow==23.0.1
pycparser==2.23
Pygments==2.19.2
pyparsing==3.2.3
python-dateutil==2.9.0.post0
python-json-logger==4.0.0
pytz==2025.2
PyYAML==6.0.3
pyzmq==27.1.0
referencing==0.37.0
requests==2.32.5
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rpds-py==0.30.0
scikit-learn==1.8.0
scipy==1.16.0
Send2Trash==2.1.0
setuptools==81.0.0
shap==0.50.0
six==1.17.0
slicer==0.0.8
soupsieve==2.8.2
SQLAlchemy==2.0.46
stack-data==0.6.3
sympy==1.14.0
terminado==0.18.1
threadpoolctl==3.6.0
tinycss2==1.4.0
torch==2.10.0
torchvision==0.25.0
tornado==6.5.4
tqdm==4.67.3
traitlets==5.14.3
typing_extensions==4.15.0
tzdata==2025.2
uri-template==1.3.0
urllib3==2.6.3
wcwidth==0.6.0
webcolors==25.10.0
webencodings==0.5.1
websocket-client==1.9.0
websockets==16.0
wheel==0.46.3
xgboost==3.2.0
yfinance==1.0
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## License

MIT License — see `LICENSE` for details.