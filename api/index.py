from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from demo import create_demo_data, calculate_spectral_indices

app = FastAPI(title="Forest Fire Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Forest Fire Detection API",
        "endpoints": {
            "analyze": "/api/analyze",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/analyze")
async def analyze():
    try:
        # Create demo data
        bands = create_demo_data()
        
        # Calculate indices
        indices = calculate_spectral_indices(bands)
        
        # Convert numpy arrays to lists for JSON serialization
        return {
            "status": "success",
            "data": {
                "bands": {k: v.tolist() for k, v in bands.items()},
                "indices": {k: v.tolist() for k, v in indices.items()}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
