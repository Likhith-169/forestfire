"""
Test module for spectral indices calculations.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from detection.spectral_indices import SpectralIndices


class TestSpectralIndices(unittest.TestCase):
    """Test cases for spectral indices calculations."""
    
    def setUp(self):
        """Set up test data."""
        self.indices = SpectralIndices()
        
        # Create test band data
        self.nir = np.array([[0.8, 0.7], [0.9, 0.6]])
        self.red = np.array([[0.3, 0.4], [0.2, 0.5]])
        self.swir1 = np.array([[0.5, 0.6], [0.4, 0.7]])
        self.swir2 = np.array([[0.4, 0.5], [0.3, 0.6]])
        self.blue = np.array([[0.2, 0.3], [0.1, 0.4]])
    
    def test_normalize_burn_ratio(self):
        """Test NBR calculation."""
        nbr = self.indices.normalize_burn_ratio(self.nir, self.swir2)
        
        # Check that NBR is calculated correctly
        expected = (self.nir - self.swir2) / (self.nir + self.swir2 + 1e-6)
        np.testing.assert_array_almost_equal(nbr, expected)
        
        # Check that NBR values are in reasonable range
        self.assertTrue(np.all(nbr >= -1))
        self.assertTrue(np.all(nbr <= 1))
    
    def test_differenced_nbr(self):
        """Test dNBR calculation."""
        nbr_pre = np.array([[0.5, 0.3], [0.6, 0.4]])
        nbr_post = np.array([[0.2, 0.1], [0.3, 0.2]])
        
        dnbr = self.indices.differenced_nbr(nbr_pre, nbr_post)
        
        # Check that dNBR is calculated correctly
        expected = nbr_pre - nbr_post
        np.testing.assert_array_almost_equal(dnbr, expected)
    
    def test_burn_area_index(self):
        """Test BAI calculation."""
        bai = self.indices.burn_area_index(self.red, self.nir)
        
        # Check that BAI is positive
        self.assertTrue(np.all(bai > 0))
        
        # Check that BAI is finite
        self.assertTrue(np.all(np.isfinite(bai)))
    
    def test_normalized_difference_vegetation_index(self):
        """Test NDVI calculation."""
        ndvi = self.indices.normalized_difference_vegetation_index(self.nir, self.red)
        
        # Check that NDVI is calculated correctly
        expected = (self.nir - self.red) / (self.nir + self.red + 1e-6)
        np.testing.assert_array_almost_equal(ndvi, expected)
        
        # Check that NDVI values are in reasonable range
        self.assertTrue(np.all(ndvi >= -1))
        self.assertTrue(np.all(ndvi <= 1))
    
    def test_normalized_difference_moisture_index(self):
        """Test NDMI calculation."""
        ndmi = self.indices.normalized_difference_moisture_index(self.nir, self.swir1)
        
        # Check that NDMI is calculated correctly
        expected = (self.nir - self.swir1) / (self.nir + self.swir1 + 1e-6)
        np.testing.assert_array_almost_equal(ndmi, expected)
        
        # Check that NDMI values are in reasonable range
        self.assertTrue(np.all(ndmi >= -1))
        self.assertTrue(np.all(ndmi <= 1))
    
    def test_calculate_all_indices(self):
        """Test calculation of all indices."""
        bands = {
            'red': self.red,
            'nir': self.nir,
            'swir1': self.swir1,
            'swir2': self.swir2,
            'blue': self.blue
        }
        
        indices = self.indices.calculate_all_indices(bands)
        
        # Check that all expected indices are present
        expected_indices = ['ndvi', 'ndmi', 'nbr', 'bai']
        for index_name in expected_indices:
            self.assertIn(index_name, indices)
            self.assertIsInstance(indices[index_name], np.ndarray)
    
    def test_classify_burn_severity(self):
        """Test burn severity classification."""
        # Create test dNBR values
        dnbr = np.array([[-0.2, 0.1], [0.5, 0.8]])
        
        severity = self.indices.classify_burn_severity(dnbr)
        
        # Check that severity values are integers
        self.assertTrue(np.all(np.issubdtype(severity.dtype, np.integer)))
        
        # Check that severity values are in valid range
        self.assertTrue(np.all(severity >= 0))
        self.assertTrue(np.all(severity <= 4))
    
    def test_division_by_zero_handling(self):
        """Test that division by zero is handled properly."""
        # Create bands with zeros
        zero_nir = np.zeros((2, 2))
        zero_red = np.zeros((2, 2))
        
        # These should not raise errors
        try:
            ndvi = self.indices.normalized_difference_vegetation_index(zero_nir, zero_red)
            self.assertTrue(np.all(np.isfinite(ndvi)))
        except Exception as e:
            self.fail(f"NBR calculation failed with zero values: {e}")


if __name__ == '__main__':
    unittest.main()
