"""
Forest Fire Detection Module

This module implements the core fire detection algorithm that combines
thermal hotspot detection with optical confirmation using spectral indices.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from rasterio.features import shapes
from rasterio.transform import from_bounds
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from datetime import datetime, timedelta
# Import logger with fallback
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

from .spectral_indices import SpectralIndices, validate_band_data


class FireDetector:
    """
    Main fire detection class that combines thermal and optical analysis.
    
    This class implements a multi-stage fire detection pipeline:
    1. Thermal hotspot detection (VIIRS/MODIS)
    2. Optical confirmation (Sentinel-2/Landsat)
    3. Spectral index analysis
    4. Burn area delineation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the FireDetector with configuration parameters.
        
        Args:
            config: Configuration dictionary containing detection parameters
        """
        self.config = config
        self.spectral_indices = SpectralIndices()
        
        # Extract detection parameters
        self.thermal_config = config.get('detection', {}).get('thermal', {})
        self.optical_config = config.get('detection', {}).get('optical', {})
        self.spatial_config = config.get('detection', {}).get('spatial', {})
        self.temporal_config = config.get('detection', {}).get('temporal', {})
        
        # Thresholds
        self.nbr_threshold = self.optical_config.get('nbr_threshold', 0.1)
        self.dnbr_threshold = self.optical_config.get('dnbr_threshold', -0.2)
        self.bai_threshold = self.optical_config.get('bai_threshold', 0.3)
        self.min_burn_area = self.spatial_config.get('min_burn_area', 10000)
        self.buffer_distance = self.spatial_config.get('buffer_distance', 2000)
        
    def detect_thermal_hotspots(self, thermal_data: np.ndarray, 
                               brightness_temp: np.ndarray,
                               mir_band: Optional[np.ndarray] = None,
                               nir_band: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect thermal hotspots using brightness temperature and MIR/NIR ratio.
        
        Args:
            thermal_data: Thermal infrared band data
            brightness_temp: Brightness temperature array (Kelvin)
            mir_band: Middle infrared band (optional)
            nir_band: Near infrared band (optional)
            
        Returns:
            Boolean mask indicating potential fire hotspots
        """
        # Basic thermal threshold
        bt_threshold = self.thermal_config.get('brightness_temp_threshold', 320)
        thermal_mask = brightness_temp > bt_threshold
        
        # MIR/NIR ratio test (if bands available)
        if mir_band is not None and nir_band is not None:
            mir_nir_ratio = mir_band / (nir_band + 1e-6)
            ratio_threshold = self.thermal_config.get('mir_nir_ratio_threshold', 0.8)
            ratio_mask = mir_nir_ratio > ratio_threshold
            
            # Combine thermal and ratio masks
            hotspot_mask = thermal_mask & ratio_mask
        else:
            hotspot_mask = thermal_mask
            
        # Apply morphological operations to clean up noise
        from scipy import ndimage
        hotspot_mask = ndimage.binary_opening(hotspot_mask, structure=np.ones((3, 3)))
        hotspot_mask = ndimage.binary_closing(hotspot_mask, structure=np.ones((5, 5)))
        
        return hotspot_mask
    
    def confirm_with_optical_data(self, bands: Dict[str, np.ndarray],
                                 cloud_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Confirm fire detection using optical bands and spectral indices.
        
        Args:
            bands: Dictionary of optical bands (red, nir, swir1, swir2, blue)
            cloud_mask: Cloud mask (True = cloudy, False = clear)
            
        Returns:
            Dictionary containing detection masks and indices
        """
        # Validate input bands
        if not validate_band_data(bands):
            raise ValueError("Invalid band data provided")
        
        # Calculate spectral indices
        indices = self.spectral_indices.calculate_all_indices(bands)
        
        # Create detection masks based on thresholds
        masks = {}
        
        # NBR-based detection
        if 'nbr' in indices:
            masks['nbr_mask'] = indices['nbr'] < self.nbr_threshold
        
        # BAI-based detection
        if 'bai' in indices:
            masks['bai_mask'] = indices['bai'] > self.bai_threshold
        
        # NDVI-based detection (low vegetation)
        if 'ndvi' in indices:
            masks['ndvi_mask'] = indices['ndvi'] < 0.2
        
        # Combine masks
        combined_mask = np.ones_like(list(bands.values())[0], dtype=bool)
        for mask in masks.values():
            combined_mask &= mask
        
        # Apply cloud mask if provided
        if cloud_mask is not None:
            combined_mask &= ~cloud_mask
        
        # Clean up small noise
        from scipy import ndimage
        combined_mask = ndimage.binary_opening(combined_mask, structure=np.ones((3, 3)))
        
        masks['combined_mask'] = combined_mask
        masks['indices'] = indices
        
        return masks
    
    def calculate_dnbr(self, pre_fire_bands: Dict[str, np.ndarray],
                      post_fire_bands: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate differenced NBR between pre-fire and post-fire images.
        
        Args:
            pre_fire_bands: Pre-fire optical bands
            post_fire_bands: Post-fire optical bands
            
        Returns:
            dNBR array
        """
        # Calculate NBR for both images
        pre_nbr = self.spectral_indices.normalize_burn_ratio(
            pre_fire_bands['nir'], pre_fire_bands['swir2']
        )
        post_nbr = self.spectral_indices.normalize_burn_ratio(
            post_fire_bands['nir'], post_fire_bands['swir2']
        )
        
        # Calculate dNBR
        dnbr = self.spectral_indices.differenced_nbr(pre_nbr, post_nbr)
        
        return dnbr
    
    def delineate_burn_area(self, detection_mask: np.ndarray,
                           transform: Tuple[float, float, float, float, float, float],
                           crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Convert detection mask to burn area polygons.
        
        Args:
            detection_mask: Boolean mask of detected burn areas
            transform: Raster transform parameters
            crs: Coordinate reference system
            
        Returns:
            GeoDataFrame containing burn area polygons
        """
        # Convert mask to polygons
        burn_polygons = []
        
        for geom, value in shapes(detection_mask.astype(np.uint8), 
                                transform=transform):
            if value == 1:  # Burned pixel
                polygon = Polygon(geom['coordinates'][0])
                if polygon.area > self.min_burn_area:  # Filter small areas
                    burn_polygons.append(polygon)
        
        # Create GeoDataFrame
        if burn_polygons:
            gdf = gpd.GeoDataFrame(geometry=burn_polygons, crs=crs)
            
            # Calculate area and other properties
            gdf['area_m2'] = gdf.geometry.area
            gdf['area_ha'] = gdf['area_m2'] / 10000
            
            # Filter by minimum area
            gdf = gdf[gdf['area_m2'] >= self.min_burn_area]
            
            return gdf
        else:
            return gpd.GeoDataFrame(geometry=[], crs=crs)
    
    def detect_fire_events(self, 
                          thermal_data: Dict[str, np.ndarray],
                          optical_data: Dict[str, np.ndarray],
                          pre_fire_data: Optional[Dict[str, np.ndarray]] = None,
                          cloud_mask: Optional[np.ndarray] = None,
                          metadata: Optional[Dict] = None) -> Dict:
        """
        Main fire detection pipeline combining thermal and optical analysis.
        
        Args:
            thermal_data: Dictionary containing thermal bands and brightness temperature
            optical_data: Dictionary containing optical bands
            pre_fire_data: Pre-fire optical data for dNBR calculation (optional)
            cloud_mask: Cloud mask for optical data
            metadata: Additional metadata (timestamps, location, etc.)
            
        Returns:
            Dictionary containing detection results
        """
        results = {
            'timestamp': datetime.now(),
            'metadata': metadata or {},
            'detections': [],
            'summary': {}
        }
        
        try:
            # Step 1: Thermal hotspot detection
            logger.info("Detecting thermal hotspots...")
            hotspot_mask = self.detect_thermal_hotspots(
                thermal_data.get('thermal', np.zeros((100, 100))),
                thermal_data.get('brightness_temp', np.zeros((100, 100))),
                thermal_data.get('mir'),
                thermal_data.get('nir')
            )
            
            # Step 2: Optical confirmation
            logger.info("Confirming with optical data...")
            optical_results = self.confirm_with_optical_data(optical_data, cloud_mask)
            
            # Step 3: Calculate dNBR if pre-fire data available
            dnbr = None
            if pre_fire_data is not None:
                logger.info("Calculating dNBR...")
                dnbr = self.calculate_dnbr(pre_fire_data, optical_data)
                
                # Apply dNBR threshold
                dnbr_mask = dnbr < self.dnbr_threshold
                optical_results['combined_mask'] &= dnbr_mask
                optical_results['indices']['dnbr'] = dnbr
            
            # Step 4: Delineate burn areas
            logger.info("Delineating burn areas...")
            transform = metadata.get('transform', (1, 0, 0, 0, 1, 0))
            crs = metadata.get('crs', 'EPSG:4326')
            
            burn_areas = self.delineate_burn_area(
                optical_results['combined_mask'],
                transform,
                crs
            )
            
            # Step 5: Calculate confidence scores
            confidence_scores = self.calculate_confidence_scores(
                hotspot_mask, optical_results, burn_areas, metadata
            )
            
            # Step 6: Compile results
            results['detections'] = self.compile_detections(
                burn_areas, confidence_scores, optical_results['indices'], metadata
            )
            
            # Step 7: Generate summary statistics
            results['summary'] = self.generate_summary(results['detections'])
            
            logger.info(f"Detection complete: {len(results['detections'])} fire events found")
            
        except Exception as e:
            logger.error(f"Error in fire detection pipeline: {e}")
            results['error'] = str(e)
        
        return results
    
    def calculate_confidence_scores(self, 
                                   hotspot_mask: np.ndarray,
                                   optical_results: Dict,
                                   burn_areas: gpd.GeoDataFrame,
                                   metadata: Optional[Dict]) -> List[float]:
        """
        Calculate confidence scores for detected fire events.
        
        Args:
            hotspot_mask: Thermal hotspot detection mask
            optical_results: Results from optical confirmation
            burn_areas: GeoDataFrame of burn areas
            metadata: Additional metadata
            
        Returns:
            List of confidence scores (0-1)
        """
        confidence_scores = []
        
        for idx, burn_area in burn_areas.iterrows():
            # Base confidence from thermal detection
            thermal_confidence = 0.5 if np.any(hotspot_mask) else 0.1
            
            # Optical confirmation confidence
            optical_confidence = 0.0
            if 'nbr' in optical_results['indices']:
                nbr_values = optical_results['indices']['nbr']
                low_nbr_ratio = np.sum(nbr_values < self.nbr_threshold) / nbr_values.size
                optical_confidence += low_nbr_ratio * 0.3
            
            if 'bai' in optical_results['indices']:
                bai_values = optical_results['indices']['bai']
                high_bai_ratio = np.sum(bai_values > self.bai_threshold) / bai_values.size
                optical_confidence += high_bai_ratio * 0.3
            
            # dNBR confidence
            dnbr_confidence = 0.0
            if 'dnbr' in optical_results['indices']:
                dnbr_values = optical_results['indices']['dnbr']
                dnbr_confidence = np.sum(dnbr_values < self.dnbr_threshold) / dnbr_values.size * 0.4
            
            # Area-based confidence
            area_confidence = min(burn_area['area_ha'] / 100, 1.0) * 0.2
            
            # Combine confidences
            total_confidence = thermal_confidence + optical_confidence + dnbr_confidence + area_confidence
            confidence_scores.append(min(total_confidence, 1.0))
        
        return confidence_scores
    
    def compile_detections(self, 
                          burn_areas: gpd.GeoDataFrame,
                          confidence_scores: List[float],
                          indices: Dict[str, np.ndarray],
                          metadata: Optional[Dict]) -> List[Dict]:
        """
        Compile detection results into structured format.
        
        Args:
            burn_areas: GeoDataFrame of burn areas
            confidence_scores: List of confidence scores
            indices: Spectral indices
            metadata: Additional metadata
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        for idx, (_, burn_area) in enumerate(burn_areas.iterrows()):
            detection = {
                'id': f"fire_{idx}_{int(datetime.now().timestamp())}",
                'geometry': burn_area.geometry,
                'area_m2': burn_area['area_m2'],
                'area_ha': burn_area['area_ha'],
                'confidence': confidence_scores[idx] if idx < len(confidence_scores) else 0.5,
                'timestamp': metadata.get('timestamp', datetime.now()),
                'location': metadata.get('location', 'Unknown'),
                'indices': {
                    'nbr_mean': float(np.mean(indices.get('nbr', [0]))) if 'nbr' in indices else None,
                    'bai_mean': float(np.mean(indices.get('bai', [0]))) if 'bai' in indices else None,
                    'ndvi_mean': float(np.mean(indices.get('ndvi', [0]))) if 'ndvi' in indices else None,
                    'dnbr_mean': float(np.mean(indices.get('dnbr', [0]))) if 'dnbr' in indices else None,
                },
                'metadata': metadata or {}
            }
            detections.append(detection)
        
        return detections
    
    def generate_summary(self, detections: List[Dict]) -> Dict:
        """
        Generate summary statistics for detected fire events.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Summary statistics dictionary
        """
        if not detections:
            return {
                'total_events': 0,
                'total_area_ha': 0,
                'mean_confidence': 0,
                'severity_distribution': {}
            }
        
        total_events = len(detections)
        total_area_ha = sum(d['area_ha'] for d in detections)
        mean_confidence = np.mean([d['confidence'] for d in detections])
        
        # Severity distribution based on dNBR
        severity_counts = {'low': 0, 'moderate': 0, 'high': 0}
        for detection in detections:
            dnbr_mean = detection['indices'].get('dnbr_mean', 0)
            if dnbr_mean < -0.1:
                severity_counts['low'] += 1
            elif dnbr_mean < 0.44:
                severity_counts['moderate'] += 1
            else:
                severity_counts['high'] += 1
        
        return {
            'total_events': total_events,
            'total_area_ha': total_area_ha,
            'mean_confidence': mean_confidence,
            'severity_distribution': severity_counts,
            'largest_event_ha': max(d['area_ha'] for d in detections) if detections else 0
        }
