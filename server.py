# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# 1. Initialize the FastAPI app
app = FastAPI(title="California Housing Prediction API", 
              description="A high-performance production API for real estate valuation.")

# 2. Define the input data schema (Logical Safeguards)
class HousingDistrict(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

# 3. Load the pre-trained model on startup
# We do this globally so it only loads once into memory
MODEL_PATH = "models/housing_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model at {MODEL_PATH}. Did you run main.py?")

# 4. Define the Prediction Endpoint
@app.post("/predict")
async def predict_price(data: HousingDistrict):
    """
    Takes a single district description and returns a predicted median house value.
    """
    try:
        # Convert Pydantic model to a DataFrame (what the pipeline expects)
        input_df = pd.DataFrame([data.model_dump()])
        
        # Run prediction through the full pipeline (Preprocessing + XGBoost)
        prediction = model.predict(input_df)[0]
        
        return {
            "predicted_median_house_value": float(round(prediction, 2)),
            "currency": "USD"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Health Check Endpoint (Professional Standard)
@app.get("/health")
def health_check():
    return {"status": "online", "model": "XGBoost v1.0"}

if __name__ == "__main__":
    # Run the server on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

