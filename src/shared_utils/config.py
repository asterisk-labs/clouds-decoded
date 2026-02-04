from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseProcessorConfig(BaseModel):
    """
    Base configuration class for all processors.
    Utilizes Pydantic for validation and type checking.
    """
    output_dir: Optional[str] = Field(None, description="Directory to save outputs")
    n_workers: int = Field(1, description="Number of parallel workers where applicable")
    
    # Common processing flags could go here
    debug_mode: bool = False

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None):
        """Loads configuration from a YAML file. If path is None, returns default config."""
        if config_path is None:
            return cls()

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # If the yaml is empty
        if data is None:
            data = {}
            
        return cls(**data)
