"""
Spectral Indices for Forest Fire Detection

This module implements various spectral indices used for detecting and analyzing
forest fires from satellite imagery, including NBR, dNBR, BAI, and others.
"""

import numpy as np
from typing import Tuple, Optional, Union
# Optional logger (fallback to stdlib logging if loguru is unavailable)
try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("spectral_indices")

# Optional geospatial dependencies (not required for demo)
try:
    import rasterio  # type: ignore
    from rasterio.warp import reproject, Resampling  # type: ignore
    from rasterio.transform import from_bounds  # type: ignore
    RASTERIO_AVAILABLE = True
except Exception:
    RASTERIO_AVAILABLE = False
    rasterio = None  # type: ignore
    reproject = None  # type: ignore
    Resampling = None  # type: ignore
    from_bounds = None  # type: ignore


class SpectralIndices:
    """
    Calculate spectral indices for forest fire detection and analysis.
    
    This class provides methods to compute various vegetation and burn indices
    from satellite imagery bands, particularly useful for fire detection and
    burn severity assessment.
    """
    
    def __init__(self, eps: float = 1e-6):
        """
        Initialize the SpectralIndices calculator.
        
        Args:
            eps: Small epsilon value to prevent division by zero
        """
        self.eps = eps
    
    def normalize_burn_ratio(self, nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Burn Ratio (NBR).
        
        NBR = (NIR - SWIR2) / (NIR + SWIR2)
        
        NBR is sensitive to changes in vegetation moisture content and is
        commonly used to identify burned areas. Fresh burns typically have
        low NBR values.
        
        Args:
            nir: Near-infrared band (e.g., Sentinel-2 B8, Landsat B5)
            swir2: Short-wave infrared 2 band (e.g., Sentinel-2 B12, Landsat B7)
            
        Returns:
            NBR array with values typically between -1 and 1
        """
        return (nir - swir2) / (nir + swir2 + self.eps)
    
    def differenced_nbr(self, nbr_pre: np.ndarray, nbr_post: np.ndarray) -> np.ndarray:
        """
        Calculate Differenced Normalized Burn Ratio (dNBR).
        
        dNBR = NBR_pre - NBR_post
        
        dNBR measures the change in NBR between pre-fire and post-fire images.
        Higher dNBR values indicate more severe burns.
        
        Args:
            nbr_pre: NBR from pre-fire image
            nbr_post: NBR from post-fire image
            
        Returns:
            dNBR array with values typically between -2 and 2
        """
        return nbr_pre - nbr_post
    
    def burn_area_index(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate Burn Area Index (BAI).
        
        BAI = 1 / ((0.1 - RED)^2 + (0.06 - NIR)^2)
        
        BAI is designed to identify burned areas by highlighting pixels with
        low NIR and low red reflectance, which are characteristic of burned areas.
        
        Args:
            red: Red band (e.g., Sentinel-2 B4, Landsat B4)
            nir: Near-infrared band (e.g., Sentinel-2 B8, Landsat B5)
            
        Returns:
            BAI array with higher values indicating burned areas
        """
        return 1 / ((0.1 - red)**2 + (0.06 - nir)**2 + self.eps)
    
    def normalized_difference_vegetation_index(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index (NDVI).
        
        NDVI = (NIR - RED) / (NIR + RED)
        
        NDVI is a measure of vegetation health and density. Burned areas
        typically show reduced NDVI values.
        
        Args:
            nir: Near-infrared band
            red: Red band
            
        Returns:
            NDVI array with values between -1 and 1
        """
        return (nir - red) / (nir + red + self.eps)
    
    def normalized_difference_moisture_index(self, nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Moisture Index (NDMI).
        
        NDMI = (NIR - SWIR1) / (NIR + SWIR1)
        
        NDMI is sensitive to vegetation water content and can help identify
        areas with low moisture that are more susceptible to fire.
        
        Args:
            nir: Near-infrared band
            swir1: Short-wave infrared 1 band (e.g., Sentinel-2 B11, Landsat B6)
            
        Returns:
            NDMI array with values between -1 and 1
        """
        return (nir - swir1) / (nir + swir1 + self.eps)
    
    def enhanced_vegetation_index(self, nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        Calculate Enhanced Vegetation Index (EVI).
        
        EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
        
        EVI is less sensitive to atmospheric conditions than NDVI and provides
        better vegetation monitoring in high biomass regions.
        
        Args:
            nir: Near-infrared band
            red: Red band
            blue: Blue band (e.g., Sentinel-2 B2, Landsat B2)
            
        Returns:
            EVI array
        """
        return 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + self.eps)
    
    def soil_adjusted_vegetation_index(self, nir: np.ndarray, red: np.ndarray, l: float = 0.5) -> np.ndarray:
        """
        Calculate Soil Adjusted Vegetation Index (SAVI).
        
        SAVI = (1 + L) * (NIR - RED) / (NIR + RED + L)
        
        SAVI minimizes soil brightness influences and is useful in areas with
        sparse vegetation.
        
        Args:
            nir: Near-infrared band
            red: Red band
            l: Soil brightness correction factor (typically 0.5)
            
        Returns:
            SAVI array
        """
        return (1 + l) * (nir - red) / (nir + red + l + self.eps)
    
    def global_environmental_monitoring_index(self, nir: np.ndarray, swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
        """
        Calculate Global Environmental Monitoring Index (GEMI).
        
        GEMI = eta * (1 - 0.25*eta) - (RED - 0.125) / (1 - RED)
        where eta = (2 * (NIR^2 - RED^2) + 1.5*NIR + 0.5*RED) / (NIR + RED + 0.5)
        
        GEMI is less sensitive to atmospheric effects and is useful for
        vegetation monitoring in arid and semi-arid regions.
        
        Args:
            nir: Near-infrared band
            red: Red band
            swir1: Short-wave infrared 1 band
            swir2: Short-wave infrared 2 band
            
        Returns:
            GEMI array
        """
        # Note: This is a simplified version. Full GEMI calculation is more complex
        eta = (2 * (nir**2 - swir1**2) + 1.5*nir + 0.5*swir1) / (nir + swir1 + 0.5 + self.eps)
        return eta * (1 - 0.25*eta) - (swir1 - 0.125) / (1 - swir1 + self.eps)
    
    def calculate_all_indices(self, bands: dict) -> dict:
        """
        Calculate all relevant spectral indices from input bands.
        
        Args:
            bands: Dictionary containing band arrays with keys like 'red', 'nir', 'swir1', 'swir2', 'blue'
            
        Returns:
            Dictionary containing all calculated indices
        """
        indices = {}
        
        # Basic indices
        if 'nir' in bands and 'red' in bands:
            indices['ndvi'] = self.normalized_difference_vegetation_index(bands['nir'], bands['red'])
        
        if 'nir' in bands and 'swir1' in bands:
            indices['ndmi'] = self.normalized_difference_moisture_index(bands['nir'], bands['swir1'])
        
        if 'nir' in bands and 'swir2' in bands:
            indices['nbr'] = self.normalize_burn_ratio(bands['nir'], bands['swir2'])
        
        if 'red' in bands and 'nir' in bands:
            indices['bai'] = self.burn_area_index(bands['red'], bands['nir'])
        
        # Enhanced indices
        if all(b in bands for b in ['nir', 'red', 'blue']):
            indices['evi'] = self.enhanced_vegetation_index(bands['nir'], bands['red'], bands['blue'])
        
        if 'nir' in bands and 'red' in bands:
            indices['savi'] = self.soil_adjusted_vegetation_index(bands['nir'], bands['red'])
        
        return indices
    
    def classify_burn_severity(self, dnbr: np.ndarray) -> np.ndarray:
        """
        Classify burn severity based on dNBR values.
        
        Classification based on USGS standards:
        - Unburned: dNBR < -0.1
        - Low severity: -0.1 <= dNBR < 0.27
        - Moderate-low severity: 0.27 <= dNBR < 0.44
        - Moderate-high severity: 0.44 <= dNBR < 0.66
        - High severity: dNBR >= 0.66
        
        Args:
            dnbr: Differenced NBR array
            
        Returns:
            Classification array (0=unburned, 1=low, 2=moderate-low, 3=moderate-high, 4=high)
        """
        severity = np.zeros_like(dnbr, dtype=np.uint8)
        
        # Apply thresholds
        severity[(dnbr >= -0.1) & (dnbr < 0.27)] = 1  # Low
        severity[(dnbr >= 0.27) & (dnbr < 0.44)] = 2  # Moderate-low
        severity[(dnbr >= 0.44) & (dnbr < 0.66)] = 3  # Moderate-high
        severity[dnbr >= 0.66] = 4  # High
        
        return severity


def resample_bands_to_match(bands: dict, target_resolution: float, 
                           target_crs: str = "EPSG:4326") -> dict:
    """
    Resample all bands to match the same resolution and CRS.
    
    Args:
        bands: Dictionary of band arrays with their metadata
        target_resolution: Target resolution in meters
        target_crs: Target coordinate reference system
        
    Returns:
        Dictionary of resampled bands
    """
    if not RASTERIO_AVAILABLE:
        logger.warning("rasterio not installed; skipping resampling and returning original bands")
        return bands

    resampled_bands = {}
    
    # Find the band with the highest resolution as reference
    reference_band = None
    reference_resolution = float('inf')
    
    for band_name, band_data in bands.items():
        if hasattr(band_data, 'res') and band_data.res[0] < reference_resolution:
            reference_band = band_name
            reference_resolution = band_data.res[0]
    
    if reference_band is None:
        logger.warning("No reference band found for resampling")
        return bands
    
    # Resample all bands to match the reference
    for band_name, band_data in bands.items():
        if hasattr(band_data, 'res') and band_data.res[0] != reference_resolution:
            # Perform resampling
            if reproject is None or from_bounds is None or Resampling is None or rasterio is None:
                logger.warning("Resampling tools unavailable; returning original band for %s", band_name)
                resampled_bands[band_name] = band_data
            else:
                try:
                    destination = from_bounds(
                        *band_data.bounds,
                        width=int(band_data.width * band_data.res[0] / reference_resolution),
                        height=int(band_data.height * band_data.res[0] / reference_resolution)
                    )
                    resampled = reproject(
                        source=band_data,
                        destination=destination,  # type: ignore[arg-type]
                        resampling=Resampling.bilinear
                    )
                    resampled_bands[band_name] = resampled[0]
                except Exception as e:
                    logger.error(f"Failed to resample band {band_name}: {e}")
                    resampled_bands[band_name] = band_data
        else:
            resampled_bands[band_name] = band_data
    
    return resampled_bands


def validate_band_data(bands: dict) -> bool:
    """
    Validate that all bands have compatible dimensions and data types.
    
    Args:
        bands: Dictionary of band arrays
        
    Returns:
        True if bands are valid, False otherwise
    """
    if not bands:
        logger.error("No bands provided for validation")
        return False
    
    # Check that all bands have the same shape
    shapes = [band.shape for band in bands.values()]
    if len(set(shapes)) > 1:
        logger.error(f"Bands have different shapes: {shapes}")
        return False
    
    # Check for invalid values
    for band_name, band_data in bands.items():
        if np.any(np.isnan(band_data)) or np.any(np.isinf(band_data)):
            logger.warning(f"Band {band_name} contains NaN or infinite values")
        
        if band_data.min() < 0 or band_data.max() > 1:
            logger.warning(f"Band {band_name} values outside expected range [0,1]")
    
    return True
