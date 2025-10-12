from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weather ML API", description="Advanced Weather Forecasting with Machine Learning", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/current")
async def get_current_weather(city: str = "London"):
    """Get current weather for a city (demo endpoint)"""
    try:
        # This would normally fetch from OpenWeatherMap API
        # For demo, return simulated current weather
        current_weather = {
            "city": city,
            "temperature": 22.5 + np.random.normal(0, 3),
            "humidity": 65 + np.random.normal(0, 10),
            "pressure": 1013.25 + np.random.normal(0, 8),
            "wind_speed": 12 + np.random.normal(0, 4),
            "description": "Partly cloudy",
            "timestamp": datetime.now().isoformat()
        }
        
        return current_weather
        
    except Exception as e:
        logger.error(f"Current weather error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Current weather error: {str(e)}")

@app.get("/model-status")
async def get_model_status():
    """Get current model status and metrics"""
    return {
        "models_trained": ml_models.is_trained,
        "model_metrics": model_metrics,
        "available_models": ["LSTM", "Random Forest", "Ensemble"],
        "training_data_size": "50,000+ samples",
        "last_updated": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)