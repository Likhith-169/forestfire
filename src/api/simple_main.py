"""
Simple FastAPI Backend for Forest Fire Detection System

This is a simplified version that works without heavy geospatial dependencies.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uvicorn
import os
from datetime import datetime, timedelta
import json
import random
import numpy as np
import asyncio

# Import logger with fallback
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class DetectionRequest(BaseModel):
    bounds: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    satellite: str = Field(default="sentinel2", description="Satellite to use")
    include_historical: bool = Field(default=True, description="Include historical data for dNBR")
    max_cloud_cover: Optional[float] = Field(default=40, description="Maximum cloud cover percentage")


class DetectionResponse(BaseModel):
    request_id: str
    status: str
    timestamp: datetime
    detections: List[Dict[str, Any]]
    summary: Dict[str, Any]
    metadata: Dict[str, Any]


class StatusResponse(BaseModel):
    request_id: str
    status: str
    progress: float
    message: str
    timestamp: datetime


# Initialize FastAPI app
app = FastAPI(
    title="Forest Fire Detection API",
    description="Satellite-based forest fire detection system",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
detection_tasks = {}  # Store ongoing detection tasks


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Forest Fire Detection API initialized successfully")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Forest Fire Detection API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "detect_fires": "/api/v1/detect",
            "get_status": "/api/v1/status/{request_id}",
            "get_results": "/api/v1/results/{request_id}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "components": {
            "api": True,
            "detection_engine": True
        }
    }


@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_fires(request: DetectionRequest, background_tasks: BackgroundTasks):
    """
    Start a fire detection analysis.
    
    This endpoint initiates a background fire detection task and returns
    a request ID for tracking progress.
    """
    try:
        # Generate request ID
        request_id = f"detection_{int(datetime.now().timestamp())}"
        
        # Validate request
        if len(request.bounds) != 4:
            raise HTTPException(status_code=400, detail="Bounds must contain 4 values: [min_lon, min_lat, max_lon, max_lat]")
        
        if request.bounds[0] >= request.bounds[2] or request.bounds[1] >= request.bounds[3]:
            raise HTTPException(status_code=400, detail="Invalid bounds: min values must be less than max values")
        
        # Initialize task status
        detection_tasks[request_id] = {
            "status": "processing",
            "progress": 0.0,
            "message": "Starting detection...",
            "timestamp": datetime.now(),
            "request": request.dict(),
            "results": None
        }
        
        # Start background task
        background_tasks.add_task(
            run_detection_task,
            request_id,
            request
        )
        
        return DetectionResponse(
            request_id=request_id,
            status="processing",
            timestamp=datetime.now(),
            detections=[],
            summary={},
            metadata={"message": "Detection task started"}
        )
        
    except Exception as e:
        logger.error(f"Error starting detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/status/{request_id}", response_model=StatusResponse)
async def get_detection_status(request_id: str):
    """Get the status of a detection task."""
    if request_id not in detection_tasks:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    task = detection_tasks[request_id]
    return StatusResponse(
        request_id=request_id,
        status=task["status"],
        progress=task["progress"],
        message=task["message"],
        timestamp=task["timestamp"]
    )


@app.get("/api/v1/results/{request_id}", response_model=DetectionResponse)
async def get_detection_results(request_id: str):
    """Get the results of a completed detection task."""
    if request_id not in detection_tasks:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    task = detection_tasks[request_id]
    
    if task["status"] == "processing":
        raise HTTPException(status_code=202, detail="Detection still in progress")
    
    if task["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Detection failed: {task.get('error', 'Unknown error')}")
    
    if task["results"] is None:
        raise HTTPException(status_code=500, detail="No results available")
    
    return DetectionResponse(
        request_id=request_id,
        status=task["status"],
        timestamp=task["timestamp"],
        detections=task["results"]["detections"],
        summary=task["results"]["summary"],
        metadata=task["results"]["metadata"]
    )


@app.get("/api/v1/hotspots")
async def get_hotspots(
    bounds: List[float] = Query(..., description="[min_lon, min_lat, max_lon, max_lat]"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    source: str = Query(default="viirs", description="Hotspot source: 'viirs' or 'modis'")
):
    """Get thermal hotspots for a given area and time period."""
    try:
        # Create deterministic seed for consistency
        seed_string = f"hotspots_{bounds}_{start_date}_{end_date}_{source}"
        seed = hash(seed_string) % (2**32)
        
        # Use deterministic pseudo-random generation
        def seeded_random(min_val, max_val, index=0):
            x = (seed + index * 12345) % 2147483647
            x = (x * 1103515245 + 12345) % 2147483647
            return min_val + (x % 1000) / 1000.0 * (max_val - min_val)
        
        # Get realistic fire zones for this area
        fire_zones = get_realistic_fire_zones(bounds)
        
        # Generate realistic number of hotspots based on area and fire zones
        area_size = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        base_hotspots = len(fire_zones) * 2 if fire_zones else 1
        max_hotspots = min(15, max(0, int(base_hotspots + area_size * 50)))
        num_hotspots = int(seeded_random(0, max_hotspots + 1, 0))
        
        hotspots = []
        
        for i in range(num_hotspots):
            # Place hotspots in realistic fire zones when possible
            if fire_zones and i < len(fire_zones) * 2:
                zone = fire_zones[i % len(fire_zones)]
                lon = zone['center'][0] + seeded_random(-0.01, 0.01, i * 2)
                lat = zone['center'][1] + seeded_random(-0.01, 0.01, i * 2 + 1)
            else:
                # Fallback to deterministic land location
                lon, lat = get_deterministic_land_location(bounds, seed, i)
            
            # Realistic hotspot properties
            confidence = get_deterministic_confidence(source, seeded_random(0, 30, i * 10), seed, i)
            brightness = get_deterministic_brightness_temp(lat, start_date, seed, i)
            frp = get_deterministic_frp(lat, start_date, seed, i)
            
            hotspot = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "id": f"hotspot_{i}_{seed}_{i}",
                    "confidence": confidence,
                    "brightness": brightness,
                    "frp": frp,  # Fire Radiative Power
                    "date": start_date,
                    "source": source,
                    "location": get_location_name(lon, lat)
                }
            }
            hotspots.append(hotspot)
        
        return {
            "type": "FeatureCollection",
            "features": hotspots,
            "metadata": {
                "seed": seed,
                "fire_zones_used": len(fire_zones),
                "area_size": area_size
            }
        }
            
    except Exception as e:
        logger.error(f"Error retrieving hotspots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/satellite-data")
async def get_satellite_data(
    bounds: List[float] = Query(..., description="[min_lon, min_lat, max_lon, max_lat]"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    satellite: str = Query(default="sentinel2", description="Satellite: 'sentinel2' or 'landsat'")
):
    """Get available satellite data for a given area and time period."""
    try:
        # Generate mock satellite scenes
        num_scenes = random.randint(1, 5)
        scenes = []
        
        for i in range(num_scenes):
            scene = {
                "id": f"{satellite}_scene_{i}",
                "satellite": satellite,
                "date": start_date,
                "cloud_cover": random.uniform(0, 40),
                "bounds": bounds,
                "bands": ["B2", "B3", "B4", "B8", "B11", "B12"] if satellite == "sentinel2" else ["B2", "B3", "B4", "B5", "B6", "B7"]
            }
            scenes.append(scene)
        
        return {
            "satellite": satellite,
            "bounds": bounds,
            "start_date": start_date,
            "end_date": end_date,
            "scenes": scenes,
            "count": len(scenes)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving satellite data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/regions")
async def get_regions():
    """Get available geographic regions."""
    regions = {
        "california": {
            "name": "California, USA",
            "bounds": [-124.5, 32.5, -114.0, 42.0],
            "center": [36.7783, -119.4179],
            "zoom": 6,
            "description": "California wildfire monitoring region"
        },
        "australia": {
            "name": "Australia",
            "bounds": [113.0, -44.0, 154.0, -10.0],
            "center": [-25.2744, 133.7751],
            "zoom": 4,
            "description": "Australian bushfire monitoring region"
        },
        "global": {
            "name": "Global Monitoring",
            "bounds": [-180.0, -90.0, 180.0, 90.0],
            "center": [20.0, 0.0],
            "zoom": 2,
            "description": "Global fire monitoring"
        }
    }
    
    return {
        "regions": regions,
        "count": len(regions)
    }


async def run_detection_task(request_id: str, request: DetectionRequest):
    """
    Background task to run fire detection analysis.
    
    This function runs the complete fire detection pipeline and updates
    the task status as it progresses.
    """
    try:
        # Create a deterministic seed based on request parameters for consistency
        # Use a more reliable deterministic hash function
        seed_string = f"{request.bounds}_{request.start_date}_{request.end_date}_{request.satellite}_{request.max_cloud_cover}"
        
        # Simple but reliable deterministic hash
        seed = 0
        for char in seed_string:
            seed = ((seed << 5) + seed + ord(char)) & 0xFFFFFFFF
        seed = seed % (2**32)  # Ensure positive seed
        
        # Set the random seed for this task
        random.seed(seed)
        
        # Update status
        detection_tasks[request_id]["progress"] = 0.1
        detection_tasks[request_id]["message"] = "Retrieving satellite data..."
        detection_tasks[request_id]["seed"] = seed  # Store seed immediately
        
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Update status
        detection_tasks[request_id]["progress"] = 0.5
        detection_tasks[request_id]["message"] = "Processing satellite data..."
        
        await asyncio.sleep(2)
        
        # Update status
        detection_tasks[request_id]["progress"] = 0.8
        detection_tasks[request_id]["message"] = "Analyzing spectral indices..."
        
        await asyncio.sleep(1)
        
        # Create mock detection results with the same seed
        mock_results = create_mock_detection_results(request, seed)
        
        # Debug logging
        logger.info(f"Mock results created: {len(mock_results.get('detections', []))} detections")
        logger.info(f"Mock metadata seed: {mock_results.get('metadata', {}).get('seed', 'N/A')}")
        logger.info(f"Task seed: {seed}")
        
        # Update status
        detection_tasks[request_id]["progress"] = 1.0
        detection_tasks[request_id]["status"] = "completed"
        detection_tasks[request_id]["message"] = "Detection completed successfully"
        detection_tasks[request_id]["results"] = mock_results
        detection_tasks[request_id]["timestamp"] = datetime.now()
        
        logger.info(f"Detection task {request_id} completed successfully with seed {seed}")
        
    except Exception as e:
        logger.error(f"Error in detection task {request_id}: {e}")
        detection_tasks[request_id]["status"] = "failed"
        detection_tasks[request_id]["error"] = str(e)
        detection_tasks[request_id]["timestamp"] = datetime.now()


def create_mock_detection_results(request: DetectionRequest, seed: int) -> Dict:
    """
    Create realistic mock detection results for demonstration purposes.
    Uses the provided seed for deterministic results.
    """
    
    # Use deterministic pseudo-random generation based on seed
    def seeded_random(min_val, max_val, index=0):
        """Generate deterministic random value based on seed and index."""
        # More robust deterministic random generation
        x = abs(seed + index * 12345) % 2147483647
        x = (x * 1103515245 + 12345) % 2147483647
        x = abs(x) % 1000000  # Ensure positive and bounded
        return min_val + (x / 1000000.0) * (max_val - min_val)
    
    def seeded_choice(items, index=0):
        """Make deterministic choice based on seed and index."""
        if not items:
            return None
        x = abs(seed + index * 12345) % 2147483647
        x = (x * 1103515245 + 12345) % 2147483647
        return items[abs(x) % len(items)]
    
    # Define realistic fire-prone areas within the bounds
    fire_zones = get_realistic_fire_zones(request.bounds)
    
    # Generate realistic number of detections based on area size - FIXED FOR CONSISTENCY
    area_size = (request.bounds[2] - request.bounds[0]) * (request.bounds[3] - request.bounds[1])
    
    # Use a more deterministic approach based on area size and seed
    # Simple but reliable deterministic hash for area
    area_hash = 0
    for char in str(area_size):
        area_hash = ((area_hash << 5) + area_hash + ord(char)) & 0xFFFFFFFF
    area_hash = area_hash % 1000
    
    seed_mod = seed % 1000
    
    # Combine area hash and seed for deterministic but varied results
    combined_value = (area_hash + seed_mod) % 1000
    
    # Map combined value to number of detections
    if combined_value < 200:  # 20% chance
        num_detections = 0
    elif combined_value < 400:  # 20% chance
        num_detections = 1
    elif combined_value < 600:  # 20% chance
        num_detections = 2
    elif combined_value < 800:  # 20% chance
        num_detections = 3
    else:  # 20% chance
        num_detections = 4
    
    # Ensure we don't exceed reasonable limits
    max_fires = min(5, max(1, int(area_size * 50)))
    num_detections = min(num_detections, max_fires)
    
    detections = []
    
    for i in range(num_detections):
        # Select from realistic fire zones
        if fire_zones and i < len(fire_zones):
            zone = fire_zones[i % len(fire_zones)]
            lon = zone['center'][0] + seeded_random(-0.005, 0.005, i * 2)
            lat = zone['center'][1] + seeded_random(-0.005, 0.005, i * 2 + 1)
        else:
            # Fallback to deterministic land location
            lon, lat = get_deterministic_land_location(request.bounds, seed, i)
        
        # Realistic fire size based on location and season
        fire_size = get_deterministic_fire_size(lat, request.start_date, seed, i)
        
        # Realistic confidence based on satellite and conditions
        confidence = get_deterministic_confidence(request.satellite, request.max_cloud_cover or 40, seed, i)
        
        # Realistic spectral indices
        indices = get_deterministic_spectral_indices(lat, request.start_date, seed, i)
        
        detection = {
            "id": f"fire_{i}_{seed}_{i}",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon - 0.005, lat - 0.005],
                    [lon + 0.005, lat - 0.005],
                    [lon + 0.005, lat + 0.005],
                    [lon - 0.005, lat + 0.005],
                    [lon - 0.005, lat - 0.005]
                ]]
            },
            "area_m2": fire_size,
            "area_ha": fire_size / 10000,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "location": get_location_name(lon, lat),
            "indices": indices,
            "metadata": {
                "satellite": request.satellite,
                "cloud_cover": seeded_random(0, request.max_cloud_cover or 40, i * 10),
                "detection_method": "mock_enhanced_deterministic"
            }
        }
        detections.append(detection)
    
    # Calculate summary
    total_area_ha = sum(d["area_ha"] for d in detections)
    mean_confidence = np.mean([d["confidence"] for d in detections]) if detections else 0
    
    summary = {
        "total_events": len(detections),
        "total_area_ha": total_area_ha,
        "mean_confidence": mean_confidence,
        "severity_distribution": {
            "low": len([d for d in detections if d["indices"]["dnbr_mean"] > -0.3]),
            "moderate": len([d for d in detections if -0.6 < d["indices"]["dnbr_mean"] <= -0.3]),
            "high": len([d for d in detections if d["indices"]["dnbr_mean"] <= -0.6])
        },
        "largest_event_ha": max(d["area_ha"] for d in detections) if detections else 0
    }
    
    return {
        "timestamp": datetime.now(),
        "metadata": {
            "request": request.dict(),
            "processing_time": seeded_random(10, 60, 999),
            "seed": seed,
            "fire_zones_used": len(fire_zones)
        },
        "detections": detections,
        "summary": summary
    }


def get_realistic_fire_zones(bounds):
    """Get realistic fire-prone zones within the given bounds."""
    zones = []
    
    # California fire zones (if bounds overlap)
    if bounds[0] <= -124.5 and bounds[2] >= -114.0 and bounds[1] <= 42.0 and bounds[3] >= 32.5:
        zones.extend([
            {"center": [-121.5, 38.5], "name": "Northern California Foothills", "risk": "high"},
            {"center": [-118.5, 34.5], "name": "Southern California Mountains", "risk": "high"},
            {"center": [-120.5, 36.5], "name": "Central Valley Edge", "risk": "medium"},
            {"center": [-116.5, 33.5], "name": "Desert Fringe", "risk": "medium"}
        ])
    
    # Australia fire zones
    if bounds[0] <= 113.0 and bounds[2] >= 154.0 and bounds[1] <= -10.0 and bounds[3] >= -44.0:
        zones.extend([
            {"center": [145.0, -37.0], "name": "Victoria Forests", "risk": "high"},
            {"center": [150.0, -33.0], "name": "New South Wales", "risk": "high"},
            {"center": [138.0, -35.0], "name": "South Australia", "risk": "medium"},
            {"center": [116.0, -32.0], "name": "Western Australia", "risk": "medium"}
        ])
    
    # Global fire zones (tropical and temperate regions)
    if bounds[2] - bounds[0] > 100:  # Large area
        zones.extend([
            {"center": [-3.5, -58.4], "name": "Amazon Basin", "risk": "high"},
            {"center": [65.0, 65.0], "name": "Siberian Taiga", "risk": "medium"},
            {"center": [20.0, 0.0], "name": "Central Africa", "risk": "high"},
            {"center": [100.0, 20.0], "name": "Southeast Asia", "risk": "medium"},
            {"center": [-120.0, 40.0], "name": "North America", "risk": "medium"},
            {"center": [135.0, -25.0], "name": "Australia", "risk": "high"}
        ])
    
    # Filter zones to only include those within bounds
    filtered_zones = []
    for zone in zones:
        if (bounds[0] <= zone["center"][0] <= bounds[2] and 
            bounds[1] <= zone["center"][1] <= bounds[3]):
            filtered_zones.append(zone)
    
    return filtered_zones


def get_land_location(bounds):
    """Get a realistic land location, avoiding major water bodies."""
    
    def is_in_water(lon, lat):
        """Check if coordinates are in water using comprehensive detection."""
        
        # 1. MAJOR OCEAN BASINS - More precise coverage
        if (lon < -130 and lat < 60):  # Pacific Ocean (far west)
            return True
        if (-80 <= lon < -60 and lat < 60):  # Atlantic Ocean (east coast)
            return True
        if (lon >= 120 and lat < 60):  # Indian Ocean (far east)
            return True
        if (lat >= 70):  # Arctic Ocean (very far north)
            return True
        
        # 2. SPECIFIC WATER BODIES
        # North America
        if (-87.5 <= lon <= -87.0 and 41.5 <= lat <= 42.5):  # Lake Michigan
            return True
        if (-122.5 <= lon <= -122.0 and 37.5 <= lat <= 38.5):  # San Francisco Bay
            return True
        if (-74.5 <= lon <= -74.0 and 40.5 <= lat <= 41.5):  # New York Bay
            return True
        if (-118.5 <= lon <= -118.0 and 33.5 <= lat <= 34.5):  # Santa Monica Bay
            return True
        
        # Australia
        if (145.0 <= lon <= 145.5 and -38.0 <= lat <= -37.5):  # Port Phillip Bay
            return True
        if (151.0 <= lon <= 151.5 and -34.0 <= lat <= -33.5):  # Botany Bay
            return True
        
        # 3. COASTAL MARGINS - Much wider to avoid near-coast areas
        coastal_margin = 0.2  # 0.2 degrees from coast (much wider)
        
        # North America coasts - Only very close to actual coastlines
        if (lon < -124 and 32 <= lat <= 42):  # California coast (very close)
            return True
        if (-75 <= lon <= -70 and 35 <= lat <= 45):  # East Coast (very close)
            return True
        
        # Australia coasts - Only very close to actual coastlines
        if (113 <= lon <= 115 and -44 <= lat <= -10):  # West Coast (very close)
            return True
        if (153 <= lon <= 155 and -44 <= lat <= -10):  # East Coast (very close)
            return True
        
        # Europe coasts - Only very close to actual coastlines
        if (-5 <= lon <= 5 and 50 <= lat <= 55):  # North Sea (very close)
            return True
        if (15 <= lon <= 25 and 35 <= lat <= 45):  # Mediterranean (very close)
            return True
        
        # Asia coasts - Only very close to actual coastlines
        if (100 <= lon <= 105 and 0 <= lat <= 15):  # Southeast Asia (very close)
            return True
        if (135 <= lon <= 140 and 35 <= lat <= 40):  # Japan (very close)
            return True
        
        # 4. MAJOR SEAS AND GULFS
        if (-5 <= lon <= 5 and 50 <= lat <= 60):  # North Sea
            return True
        if (25 <= lon <= 35 and 30 <= lat <= 45):  # Mediterranean
            return True
        if (50 <= lon <= 60 and 20 <= lat <= 30):  # Persian Gulf
            return True
        if (80 <= lon <= 90 and 5 <= lat <= 20):  # Bay of Bengal
            return True
        
        # 5. ISLAND NATIONS - Avoid placing fires on small islands
        if (lon < -170 or lon > 170):  # Very far east/west
            return True
        if (abs(lat) > 80):  # Very far north/south
            return True
        
        return False
    
    # Try to find a land location
    max_attempts = 200  # Increased attempts
    for attempt in range(max_attempts):
        lon = random.uniform(bounds[0], bounds[2])
        lat = random.uniform(bounds[1], bounds[3])
        
        if not is_in_water(lon, lat):
            return lon, lat
    
    # If random attempts fail, try systematic search from center
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    
    # Search in expanding circles from center
    for radius in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        for angle in range(0, 360, 10):  # Check every 10 degrees
            import math
            test_lon = center_lon + radius * math.cos(math.radians(angle))
            test_lat = center_lat + radius * math.sin(math.radians(angle))
            
            if (bounds[0] <= test_lon <= bounds[2] and 
                bounds[1] <= test_lat <= bounds[3] and
                not is_in_water(test_lon, test_lat)):
                return test_lon, test_lat
    
    # Last resort: return a known land location within bounds
    known_land_locations = [
        # North America
        (-120.0, 40.0),  # California
        (-80.0, 40.0),   # Pennsylvania
        (-100.0, 40.0),  # Kansas
        
        # Europe
        (0.0, 50.0),     # London area
        (10.0, 50.0),    # Germany
        (-5.0, 40.0),    # Spain
        
        # Asia
        (100.0, 40.0),   # China
        (80.0, 20.0),    # India
        (135.0, 35.0),   # Japan
        
        # Australia
        (135.0, -25.0),  # Central Australia
        (145.0, -37.0),  # Victoria
        (150.0, -33.0),  # New South Wales
    ]
    
    for land_lon, land_lat in known_land_locations:
        if (bounds[0] <= land_lon <= bounds[2] and 
            bounds[1] <= land_lat <= bounds[3]):
            return land_lon, land_lat
    
    # Absolute fallback: return center with offset
    return center_lon + 0.1, center_lat + 0.1


def get_realistic_fire_size(lat, start_date):
    """Get realistic fire size based on location and season."""
    # Parse date to get season
    try:
        date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        month = date_obj.month
    except:
        month = 6  # Default to summer
    
    # Base size varies by latitude and season
    if abs(lat) < 23.5:  # Tropical
        base_size = 50000  # 5 ha
        season_multiplier = 1.5 if month in [12, 1, 2] else 1.0  # Dry season
    elif abs(lat) < 45:  # Temperate
        base_size = 100000  # 10 ha
        season_multiplier = 2.0 if month in [6, 7, 8] else 0.5  # Summer
    else:  # Boreal
        base_size = 200000  # 20 ha
        season_multiplier = 2.5 if month in [6, 7, 8] else 0.5  # Summer
    
    # Add some realistic variation
    variation = random.uniform(0.5, 1.5)
    return base_size * season_multiplier * variation


def get_realistic_confidence(satellite, cloud_cover):
    """Get realistic confidence based on satellite and conditions."""
    # Base confidence by satellite
    base_confidence = {
        "sentinel2": 0.85,
        "landsat": 0.80
    }.get(satellite.lower(), 0.75)
    
    # Reduce confidence with cloud cover
    cloud_penalty = min(0.3, cloud_cover / 100 * 0.5)
    
    # Add some realistic variation
    variation = random.uniform(-0.1, 0.1)
    
    return max(0.5, min(0.95, base_confidence - cloud_penalty + variation))


def get_realistic_spectral_indices(lat, start_date):
    """Get realistic spectral indices based on location and season."""
    # Parse date to get season
    try:
        date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        month = date_obj.month
    except:
        month = 6  # Default to summer
    
    # Base indices vary by latitude and season
    if abs(lat) < 23.5:  # Tropical
        nbr_base = -0.3
        dnbr_base = -0.5
        bai_base = 0.6
        ndvi_base = 0.2
    elif abs(lat) < 45:  # Temperate
        nbr_base = -0.4
        dnbr_base = -0.6
        bai_base = 0.7
        ndvi_base = 0.15
    else:  # Boreal
        nbr_base = -0.5
        dnbr_base = -0.7
        bai_base = 0.8
        ndvi_base = 0.1
    
    # Seasonal adjustments
    if month in [6, 7, 8]:  # Summer
        nbr_base -= 0.1
        dnbr_base -= 0.1
        bai_base += 0.1
        ndvi_base -= 0.05
    
    # Add realistic variation
    variation = 0.1
    return {
        "nbr_mean": nbr_base + random.uniform(-variation, variation),
        "bai_mean": bai_base + random.uniform(-variation, variation),
        "ndvi_mean": ndvi_base + random.uniform(-variation, variation),
        "dnbr_mean": dnbr_base + random.uniform(-variation, variation)
    }


def get_location_name(lon, lat):
    """Get a realistic location name based on coordinates."""
    # Major cities and regions for reference
    locations = [
        {"coords": [-122.4194, 37.7749], "name": "San Francisco Area"},
        {"coords": [-118.2437, 34.0522], "name": "Los Angeles Area"},
        {"coords": [-121.4944, 38.5816], "name": "Sacramento Area"},
        {"coords": [145.0, -37.0], "name": "Victoria Region"},
        {"coords": [150.0, -33.0], "name": "New South Wales"},
        {"coords": [-3.5, -58.4], "name": "Amazon Basin"},
        {"coords": [65.0, 65.0], "name": "Siberian Region"}
    ]
    
    # Find closest location
    min_distance = float('inf')
    closest_name = "Unknown Location"
    
    for loc in locations:
        distance = ((lon - loc["coords"][0])**2 + (lat - loc["coords"][1])**2)**0.5
        if distance < min_distance:
            min_distance = distance
            closest_name = loc["name"]
    
    # Add some variation
    if min_distance < 0.1:  # Within 0.1 degrees
        return closest_name
    elif min_distance < 0.5:  # Within 0.5 degrees
        return f"Near {closest_name}"
    else:
        return f"Remote Area ({lon:.2f}, {lat:.2f})"


def get_deterministic_land_location(bounds, seed, index):
    """Get a deterministic land location, avoiding major water bodies."""
    
    def is_in_water(lon, lat):
        """Check if coordinates are in water using comprehensive detection."""
        
        # 1. MAJOR OCEAN BASINS - More precise coverage
        if (lon < -130 and lat < 60):  # Pacific Ocean (far west)
            return True
        if (-80 <= lon < -60 and lat < 60):  # Atlantic Ocean (east coast)
            return True
        if (lon >= 120 and lat < 60):  # Indian Ocean (far east)
            return True
        if (lat >= 70):  # Arctic Ocean (very far north)
            return True
        
        # 2. SPECIFIC WATER BODIES
        # North America
        if (-87.5 <= lon <= -87.0 and 41.5 <= lat <= 42.5):  # Lake Michigan
            return True
        if (-122.5 <= lon <= -122.0 and 37.5 <= lat <= 38.5):  # San Francisco Bay
            return True
        if (-74.5 <= lon <= -74.0 and 40.5 <= lat <= 41.5):  # New York Bay
            return True
        if (-118.5 <= lon <= -118.0 and 33.5 <= lat <= 34.5):  # Santa Monica Bay
            return True
        
        # Australia
        if (145.0 <= lon <= 145.5 and -38.0 <= lat <= -37.5):  # Port Phillip Bay
            return True
        if (151.0 <= lon <= 151.5 and -34.0 <= lat <= -33.5):  # Botany Bay
            return True
        
        # 3. COASTAL MARGINS - Much wider to avoid near-coast areas
        coastal_margin = 0.2  # 0.2 degrees from coast (much wider)
        
        # North America coasts - Only very close to actual coastlines
        if (lon < -124 and 32 <= lat <= 42):  # California coast (very close)
            return True
        if (-75 <= lon <= -70 and 35 <= lat <= 45):  # East Coast (very close)
            return True
        
        # Australia coasts - Only very close to actual coastlines
        if (113 <= lon <= 115 and -44 <= lat <= -10):  # West Coast (very close)
            return True
        if (153 <= lon <= 155 and -44 <= lat <= -10):  # East Coast (very close)
            return True
        
        # Europe coasts - Only very close to actual coastlines
        if (-5 <= lon <= 5 and 50 <= lat <= 55):  # North Sea (very close)
            return True
        if (15 <= lon <= 25 and 35 <= lat <= 45):  # Mediterranean (very close)
            return True
        
        # Asia coasts - Only very close to actual coastlines
        if (100 <= lon <= 105 and 0 <= lat <= 15):  # Southeast Asia (very close)
            return True
        if (135 <= lon <= 140 and 35 <= lat <= 40):  # Japan (very close)
            return True
        
        # 4. MAJOR SEAS AND GULFS
        if (-5 <= lon <= 5 and 50 <= lat <= 60):  # North Sea
            return True
        if (25 <= lon <= 35 and 30 <= lat <= 45):  # Mediterranean
            return True
        if (50 <= lon <= 60 and 20 <= lat <= 30):  # Persian Gulf
            return True
        if (80 <= lon <= 90 and 5 <= lat <= 20):  # Bay of Bengal
            return True
        
        # 5. ISLAND NATIONS - Avoid placing fires on small islands
        if (lon < -170 or lon > 170):  # Very far east/west
            return True
        if (abs(lat) > 80):  # Very far north/south
            return True
        
        return False
    
    # Use seed to generate deterministic coordinates
    def seeded_random(min_val, max_val, idx):
        x = (seed + index * 12345 + idx * 67890) % 2147483647
        x = (x * 1103515245 + 12345) % 2147483647
        return min_val + (x % 1000) / 1000.0 * (max_val - min_val)
    
    # Try deterministic locations
    for attempt in range(20):  # Increased attempts
        # Generate deterministic coordinates
        lon = seeded_random(bounds[0], bounds[2], attempt * 2)
        lat = seeded_random(bounds[1], bounds[3], attempt * 2 + 1)
        
        if not is_in_water(lon, lat):
            return lon, lat
    
    # If deterministic attempts fail, try systematic search from center
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    
    # Search in expanding circles from center
    for radius in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        for angle in range(0, 360, 15):  # Check every 15 degrees
            import math
            test_lon = center_lon + radius * math.cos(math.radians(angle))
            test_lat = center_lat + radius * math.sin(math.radians(angle))
            
            if (bounds[0] <= test_lon <= bounds[2] and 
                bounds[1] <= test_lat <= bounds[3] and
                not is_in_water(test_lon, test_lat)):
                return test_lon, test_lat
    
    # Last resort: return a known land location within bounds
    known_land_locations = [
        # North America
        (-120.0, 40.0),  # California
        (-80.0, 40.0),   # Pennsylvania
        (-100.0, 40.0),  # Kansas
        
        # Europe
        (0.0, 50.0),     # London area
        (10.0, 50.0),    # Germany
        (-5.0, 40.0),    # Spain
        
        # Asia
        (100.0, 40.0),   # China
        (80.0, 20.0),    # India
        (135.0, 35.0),   # Japan
        
        # Australia
        (135.0, -25.0),  # Central Australia
        (145.0, -37.0),  # Victoria
        (150.0, -33.0),  # New South Wales
    ]
    
    for land_lon, land_lat in known_land_locations:
        if (bounds[0] <= land_lon <= bounds[2] and 
            bounds[1] <= land_lat <= bounds[3]):
            return land_lon, land_lat
    
    # Absolute fallback: return center with deterministic offset
    offset_lon = seeded_random(-0.1, 0.1, index * 100)
    offset_lat = seeded_random(-0.1, 0.1, index * 100 + 1)
    return center_lon + offset_lon, center_lat + offset_lat


def get_deterministic_fire_size(lat, start_date, seed, index):
    """Get deterministic fire size based on location and season."""
    def seeded_random(min_val, max_val, idx):
        x = (seed + index * 12345 + idx * 67890) % 2147483647
        x = (x * 1103515245 + 12345) % 2147483647
        return min_val + (x % 1000) / 1000.0 * (max_val - min_val)
    
    # Parse date to get season
    try:
        date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        month = date_obj.month
    except:
        month = 6  # Default to summer
    
    # Base size varies by latitude and season
    if abs(lat) < 23.5:  # Tropical
        base_size = 50000  # 5 ha
        season_multiplier = 1.5 if month in [12, 1, 2] else 1.0  # Dry season
    elif abs(lat) < 45:  # Temperate
        base_size = 100000  # 10 ha
        season_multiplier = 2.0 if month in [6, 7, 8] else 0.5  # Summer
    else:  # Boreal
        base_size = 200000  # 20 ha
        season_multiplier = 2.5 if month in [6, 7, 8] else 0.5  # Summer
    
    # Add deterministic variation
    variation = seeded_random(0.5, 1.5, index * 200)
    return base_size * season_multiplier * variation


def get_deterministic_confidence(satellite, cloud_cover, seed, index):
    """Get deterministic confidence based on satellite and conditions."""
    def seeded_random(min_val, max_val, idx):
        x = (seed + index * 12345 + idx * 67890) % 2147483647
        x = (x * 1103515245 + 12345) % 2147483647
        return min_val + (x % 1000) / 1000.0 * (max_val - min_val)
    
    # Base confidence by satellite
    base_confidence = {
        "sentinel2": 0.85,
        "landsat": 0.80
    }.get(satellite.lower(), 0.75)
    
    # Reduce confidence with cloud cover
    cloud_penalty = min(0.3, cloud_cover / 100 * 0.5)
    
    # Add deterministic variation
    variation = seeded_random(-0.1, 0.1, index * 300)
    
    return max(0.5, min(0.95, base_confidence - cloud_penalty + variation))


def get_deterministic_spectral_indices(lat, start_date, seed, index):
    """Get deterministic spectral indices based on location and season."""
    def seeded_random(min_val, max_val, idx):
        x = (seed + index * 12345 + idx * 67890) % 2147483647
        x = (x * 1103515245 + 12345) % 2147483647
        return min_val + (x % 1000) / 1000.0 * (max_val - min_val)
    
    # Parse date to get season
    try:
        date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        month = date_obj.month
    except:
        month = 6  # Default to summer
    
    # Base indices vary by latitude and season
    if abs(lat) < 23.5:  # Tropical
        nbr_base = -0.3
        dnbr_base = -0.5
        bai_base = 0.6
        ndvi_base = 0.2
    elif abs(lat) < 45:  # Temperate
        nbr_base = -0.4
        dnbr_base = -0.6
        bai_base = 0.7
        ndvi_base = 0.15
    else:  # Boreal
        nbr_base = -0.5
        dnbr_base = -0.7
        bai_base = 0.8
        ndvi_base = 0.1
    
    # Seasonal adjustments
    if month in [6, 7, 8]:  # Summer
        nbr_base -= 0.1
        dnbr_base -= 0.1
        bai_base += 0.1
        ndvi_base -= 0.05
    
    # Add deterministic variation
    variation = 0.1
    return {
        "nbr_mean": nbr_base + seeded_random(-variation, variation, index * 400),
        "bai_mean": bai_base + seeded_random(-variation, variation, index * 400 + 1),
        "ndvi_mean": ndvi_base + seeded_random(-variation, variation, index * 400 + 2),
        "dnbr_mean": dnbr_base + seeded_random(-variation, variation, index * 400 + 3)
    }


def get_realistic_brightness_temp(lat, start_date):
    """Get realistic brightness temperature based on location and season."""
    try:
        date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        month = date_obj.month
    except:
        month = 6  # Default to summer
    
    # Base temperature varies by latitude and season
    if abs(lat) < 23.5:  # Tropical
        base_temp = 320  # Kelvin
        season_multiplier = 1.1 if month in [12, 1, 2] else 1.0  # Dry season
    elif abs(lat) < 45:  # Temperate
        base_temp = 330  # Kelvin
        season_multiplier = 1.2 if month in [6, 7, 8] else 0.9  # Summer
    else:  # Boreal
        base_temp = 340  # Kelvin
        season_multiplier = 1.3 if month in [6, 7, 8] else 0.8  # Summer
    
    # Add realistic variation
    variation = random.uniform(-10, 20)
    return base_temp * season_multiplier + variation


def get_realistic_frp(lat, start_date):
    """Get realistic Fire Radiative Power based on location and season."""
    try:
        date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        month = date_obj.month
    except:
        month = 6  # Default to summer
    
    # Base FRP varies by latitude and season
    if abs(lat) < 23.5:  # Tropical
        base_frp = 30  # MW
        season_multiplier = 1.5 if month in [12, 1, 2] else 1.0  # Dry season
    elif abs(lat) < 45:  # Temperate
        base_frp = 50  # MW
        season_multiplier = 2.0 if month in [6, 7, 8] else 0.5  # Summer
    else:  # Boreal
        base_frp = 80  # MW
        season_multiplier = 2.5 if month in [6, 7, 8] else 0.3  # Summer
    
    # Add realistic variation
    variation = random.uniform(0.5, 1.5)
    return base_frp * season_multiplier * variation


def get_deterministic_brightness_temp(lat, start_date, seed, index):
    """Get deterministic brightness temperature based on location and season."""
    def seeded_random(min_val, max_val, idx):
        x = (seed + index * 12345 + idx * 67890) % 2147483647
        x = (x * 1103515245 + 12345) % 2147483647
        return min_val + (x % 1000) / 1000.0 * (max_val - min_val)
    
    try:
        date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        month = date_obj.month
    except:
        month = 6  # Default to summer
    
    # Base temperature varies by latitude and season
    if abs(lat) < 23.5:  # Tropical
        base_temp = 320  # Kelvin
        season_multiplier = 1.1 if month in [12, 1, 2] else 1.0  # Dry season
    elif abs(lat) < 45:  # Temperate
        base_temp = 330  # Kelvin
        season_multiplier = 1.2 if month in [6, 7, 8] else 0.9  # Summer
    else:  # Boreal
        base_temp = 340  # Kelvin
        season_multiplier = 1.3 if month in [6, 7, 8] else 0.8  # Summer
    
    # Add deterministic variation
    variation = seeded_random(-10, 20, index * 500)
    return base_temp * season_multiplier + variation


def get_deterministic_frp(lat, start_date, seed, index):
    """Get deterministic Fire Radiative Power based on location and season."""
    def seeded_random(min_val, max_val, idx):
        x = (seed + index * 12345 + idx * 67890) % 2147483647
        x = (x * 1103515245 + 12345) % 2147483647
        return min_val + (x % 1000) / 1000.0 * (max_val - min_val)
    
    try:
        date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        month = date_obj.month
    except:
        month = 6  # Default to summer
    
    # Base FRP varies by latitude and season
    if abs(lat) < 23.5:  # Tropical
        base_frp = 30  # MW
        season_multiplier = 1.5 if month in [12, 1, 2] else 1.0  # Dry season
    elif abs(lat) < 45:  # Temperate
        base_frp = 50  # MW
        season_multiplier = 2.0 if month in [6, 7, 8] else 0.5  # Summer
    else:  # Boreal
        base_frp = 80  # MW
        season_multiplier = 2.5 if month in [6, 7, 8] else 0.3  # Summer
    
    # Add deterministic variation
    variation = seeded_random(0.5, 1.5, index * 600)
    return base_frp * season_multiplier * variation


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "simple_main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
