"""
Advanced Configuration Loader for Forest Fire Detection System

This module provides configuration loading, validation, and management
with support for environment variables, multiple formats, and validation.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import logging
import warnings

# Create a simple logger if loguru is not available
try:
from loguru import logger
except ImportError:
    # Fallback to standard logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# Configuration Models
class SystemConfig(BaseModel):
    name: str = Field(default="Advanced Forest Fire Detection System")
    version: str = Field(default="2.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    timezone: str = Field(default="UTC")

class APIConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1)
    max_connections: int = Field(default=1000, ge=1)
    timeout: int = Field(default=300, ge=1)
    rate_limit: Dict[str, int] = Field(default_factory=dict)
    cors: Dict[str, Any] = Field(default_factory=dict)

class DatabaseConfig(BaseModel):
    type: str = Field(default="postgresql")
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    name: str = Field(default="forestfire_db")
    user: str = Field(default="forestfire_user")
    password: Optional[str] = None
    pool_size: int = Field(default=20, ge=1)
    max_overflow: int = Field(default=30, ge=0)
    echo: bool = Field(default=False)

class RedisConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: Optional[str] = None
    max_connections: int = Field(default=50, ge=1)

class CeleryConfig(BaseModel):
    broker_url: str = Field(default="redis://localhost:6379/1")
    result_backend: str = Field(default="redis://localhost:6379/2")
    task_serializer: str = Field(default="json")
    result_serializer: str = Field(default="json")
    accept_content: list = Field(default=["json"])
    timezone: str = Field(default="UTC")
    enable_utc: bool = Field(default=True)
    worker_concurrency: int = Field(default=4, ge=1)
    task_acks_late: bool = Field(default=True)
    worker_prefetch_multiplier: int = Field(default=1, ge=1)

class SatelliteDataConfig(BaseModel):
    viirs: Dict[str, Any] = Field(default_factory=dict)
    modis: Dict[str, Any] = Field(default_factory=dict)
    sentinel2: Dict[str, Any] = Field(default_factory=dict)
    landsat: Dict[str, Any] = Field(default_factory=dict)
    gee: Dict[str, Any] = Field(default_factory=dict)

class DetectionConfig(BaseModel):
    thermal: Dict[str, Any] = Field(default_factory=dict)
    optical: Dict[str, Any] = Field(default_factory=dict)
    ml: Dict[str, Any] = Field(default_factory=dict)
    spatial: Dict[str, Any] = Field(default_factory=dict)
    temporal: Dict[str, Any] = Field(default_factory=dict)

class MLModelsConfig(BaseModel):
    unet: Dict[str, Any] = Field(default_factory=dict)
    transformer: Dict[str, Any] = Field(default_factory=dict)
    ensemble: Dict[str, Any] = Field(default_factory=dict)
    xgboost: Dict[str, Any] = Field(default_factory=dict)

class DataProcessingConfig(BaseModel):
    cloud_masking: Dict[str, Any] = Field(default_factory=dict)
    atmospheric_correction: Dict[str, Any] = Field(default_factory=dict)
    geometric_correction: Dict[str, Any] = Field(default_factory=dict)
    radiometric_calibration: Dict[str, Any] = Field(default_factory=dict)

class PerformanceConfig(BaseModel):
    parallel: Dict[str, Any] = Field(default_factory=dict)
    memory: Dict[str, Any] = Field(default_factory=dict)
    caching: Dict[str, Any] = Field(default_factory=dict)

class MonitoringConfig(BaseModel):
    metrics: Dict[str, Any] = Field(default_factory=dict)
    health_checks: Dict[str, Any] = Field(default_factory=dict)
    alerting: Dict[str, Any] = Field(default_factory=dict)

class SecurityConfig(BaseModel):
    authentication: Dict[str, Any] = Field(default_factory=dict)
    rate_limiting: Dict[str, Any] = Field(default_factory=dict)
    api_keys: Dict[str, Any] = Field(default_factory=dict)

class RegionConfig(BaseModel):
    name: str
    bounds: list = Field(..., min_items=4, max_items=4)
    center: list = Field(..., min_items=2, max_items=2)
    zoom: int = Field(..., ge=1, le=20)
    description: str = ""

class ExternalServicesConfig(BaseModel):
    weather: Dict[str, Any] = Field(default_factory=dict)
    emergency: Dict[str, Any] = Field(default_factory=dict)
    social_media: Dict[str, Any] = Field(default_factory=dict)

class DevelopmentConfig(BaseModel):
    mock_data: Dict[str, Any] = Field(default_factory=dict)
    testing: Dict[str, Any] = Field(default_factory=dict)
    debugging: Dict[str, Any] = Field(default_factory=dict)

class MainConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    satellite_data: SatelliteDataConfig = Field(default_factory=SatelliteDataConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    ml_models: MLModelsConfig = Field(default_factory=MLModelsConfig)
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    regions: Dict[str, RegionConfig] = Field(default_factory=dict)
    external_services: ExternalServicesConfig = Field(default_factory=ExternalServicesConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)

    @validator('regions', pre=True)
    def validate_regions(cls, v):
        if isinstance(v, dict):
            return {k: RegionConfig(**region_data) if isinstance(region_data, dict) else region_data 
                   for k, region_data in v.items()}
        return v

class ConfigLoader:
    """
    Advanced configuration loader with validation and environment variable support.
    """
    
    def __init__(self, config_path: Optional[str] = None, env_prefix: str = "FORESTFIRE"):
        """
        Initialize the configuration loader.
    
    Args:
            config_path: Path to configuration file
            env_prefix: Environment variable prefix
        """
        self.config_path = config_path
        self.env_prefix = env_prefix
        self.config: Optional[MainConfig] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment variables."""
        try:
            # Load from file if specified
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
            else:
                file_config = {}
            
            # Load from environment variables
            env_config = self._load_from_env()
            
            # Merge configurations (env overrides file)
            merged_config = self._merge_configs(file_config, env_config)
            
            # Validate configuration
            self.config = MainConfig(**merged_config)
            
            logger.info(f"Configuration loaded successfully from {self.config_path or 'defaults'}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Load default configuration
            self.config = MainConfig()
            logger.warning("Using default configuration")
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to configuration paths
        env_mappings = {
            'FORESTFIRE_ENVIRONMENT': ['system', 'environment'],
            'FORESTFIRE_DEBUG': ['system', 'debug'],
            'FORESTFIRE_LOG_LEVEL': ['system', 'log_level'],
            'FORESTFIRE_API_HOST': ['api', 'host'],
            'FORESTFIRE_API_PORT': ['api', 'port'],
            'FORESTFIRE_DB_HOST': ['database', 'host'],
            'FORESTFIRE_DB_PORT': ['database', 'port'],
            'FORESTFIRE_DB_NAME': ['database', 'name'],
            'FORESTFIRE_DB_USER': ['database', 'user'],
            'FORESTFIRE_DB_PASSWORD': ['database', 'password'],
            'FORESTFIRE_REDIS_HOST': ['redis', 'host'],
            'FORESTFIRE_REDIS_PORT': ['redis', 'port'],
            'FORESTFIRE_REDIS_PASSWORD': ['redis', 'password'],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(env_config, config_path, self._parse_env_value(value))
        
        return env_config
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try to parse as integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: Any):
        """Set a nested value in a dictionary."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _merge_configs(self, file_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge file and environment configurations."""
        merged = file_config.copy()
        
        def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dicts(base[key], value)
                else:
                    base[key] = value
        
        merge_dicts(merged, env_config)
        return merged
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if not self.config:
            return default
        
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = getattr(value, key)
            return value
        except (AttributeError, KeyError):
            return default
    
    def get_section(self, section: str) -> Optional[Dict[str, Any]]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section as dictionary
        """
        if not self.config:
            return None
        
        try:
            section_obj = getattr(self.config, section)
            return section_obj.dict() if hasattr(section_obj, 'dict') else section_obj
        except AttributeError:
            return None
    
    def validate(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            if self.config:
                # Pydantic validation is automatic
                return True
            return False
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def reload(self):
        """Reload configuration from file and environment."""
        self._load_config()
    
    def export(self, format: str = "yaml") -> str:
        """
        Export configuration to specified format.
    
    Args:
            format: Output format (yaml, json)
            
        Returns:
            Configuration as string
        """
        if not self.config:
            return ""
        
        config_dict = self.config.dict()
        
        if format.lower() == "json":
            return json.dumps(config_dict, indent=2, default=str)
        elif format.lower() == "yaml":
            return yaml.dump(config_dict, default_flow_style=False, default_representer=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        db_config = self.get_section('database')
        if not db_config:
            return ""
        
        db_type = db_config.get('type', 'postgresql')
        host = db_config.get('host', 'localhost')
        port = db_config.get('port', 5432)
        name = db_config.get('name', 'forestfire_db')
        user = db_config.get('user', 'forestfire_user')
        password = db_config.get('password', '')
        
        if db_type == 'postgresql':
            if password:
                return f"postgresql://{user}:{password}@{host}:{port}/{name}"
            else:
                return f"postgresql://{user}@{host}:{port}/{name}"
        elif db_type == 'sqlite':
            return f"sqlite:///{name}"
        else:
            return f"{db_type}://{user}:{password}@{host}:{port}/{name}"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        redis_config = self.get_section('redis')
        if not redis_config:
            return "redis://localhost:6379/0"
        
        host = redis_config.get('host', 'localhost')
        port = redis_config.get('port', 6379)
        db = redis_config.get('db', 0)
        password = redis_config.get('password')
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        else:
            return f"redis://{host}:{port}/{db}"

# Global configuration instance
_global_config: Optional[ConfigLoader] = None

def load_config(config_path: Optional[str] = None, env_prefix: str = "FORESTFIRE") -> ConfigLoader:
    """
    Load configuration globally.
    
    Args:
        config_path: Path to configuration file
        env_prefix: Environment variable prefix
        
    Returns:
        Configuration loader instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = ConfigLoader(config_path, env_prefix)
    
    return _global_config

def get_config() -> Optional[ConfigLoader]:
    """Get the global configuration instance."""
    return _global_config

def reload_config():
    """Reload the global configuration."""
    global _global_config
    if _global_config:
        _global_config.reload()

# Convenience functions
def get_setting(key_path: str, default: Any = None) -> Any:
    """Get a configuration setting by path."""
    config = get_config()
    return config.get(key_path, default) if config else default

def get_database_config() -> Optional[Dict[str, Any]]:
    """Get database configuration section."""
    config = get_config()
    return config.get_section('database') if config else None

def get_api_config() -> Optional[Dict[str, Any]]:
    """Get API configuration section."""
    config = get_config()
    return config.get_section('api') if config else None

def get_detection_config() -> Optional[Dict[str, Any]]:
    """Get detection configuration section."""
    config = get_config()
    return config.get_section('detection') if config else None
