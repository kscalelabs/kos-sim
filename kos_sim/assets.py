"""Asset management for robot models."""

import os
import subprocess
from pathlib import Path
from typing import Optional, Dict

from kos_sim import logger
from pydantic import BaseModel, Field

class JointMetadataOutput(BaseModel):
    id: Optional[int] = Field(None, title="Id")
    kp: Optional[str] = Field(None, title="Kp")
    kd: Optional[str] = Field(None, title="Kd")
    armature: Optional[str] = Field(None, title="Armature")
    friction: Optional[str] = Field(None, title="Friction")
    offset: Optional[str] = Field(None, title="Offset")
    flipped: Optional[bool] = Field(None, title="Flipped")

class RobotURDFMetadataOutput(BaseModel):
    joint_name_to_metadata: Optional[Dict[str, JointMetadataOutput]] = Field(None, title="Joint Name To Metadata")
    control_frequency: Optional[str] = Field(None, title="Control Frequency")

def get_assets_dir() -> Path:
    """Get the directory containing the assets."""
    if os.environ.get("KOS_SIM_INSTALLED"):
        return Path(__file__).parent / "kscale-assets"
    return Path(__file__).parent.parent / "kscale-assets"


def get_available_models() -> list[str]:
    """Get a list of available model names."""
    assets_dir = get_assets_dir()
    if not assets_dir.exists():
        return []
    models_dir = assets_dir / ""
    if not models_dir.exists():
        return []
    return [d.name for d in models_dir.iterdir() if d.is_dir()]


def ensure_assets_up_to_date() -> bool:
    """Ensure assets are up to date by checking and updating if necessary."""
    assets_dir = get_assets_dir()
    if not assets_dir.exists():
        logger.error("Assets directory not found at %s", assets_dir)
        return False

    try:
        # Check if we need to update
        subprocess.run(["git", "submodule", "status"], check=True, capture_output=True, cwd=Path(__file__).parent.parent)
        
        # Update the submodule
        subprocess.run(["git", "submodule", "update", "--remote", "--merge"], check=True, cwd=Path(__file__).parent.parent)
        logger.info("Assets are up to date")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to update assets: %s", e)
        return False
    except Exception as e:
        logger.error("Unexpected error updating assets: %s", e)
        return False


def get_model_path(model_name: str) -> Path:
    """Get the path to a model's assets."""
    return get_assets_dir() / model_name


def get_model_metadata(model_name: str) -> Optional[str]:
    """Get the metadata for a model."""
    model_path = get_model_path(model_name)
    metadata_path = model_path / "metadata.json"
    if not metadata_path.exists():
        logger.error("Metadata not found for model %s", model_name)
        return None
    
    try:
        import json
        data = json.loads(metadata_path.read_text())
        
        # Convert numeric values to strings to satisfy Pydantic validation
        def convert_numeric_to_string(obj):
            if isinstance(obj, dict):
                return {k: convert_numeric_to_string(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numeric_to_string(item) for item in obj]
            elif isinstance(obj, (int, float)):
                return str(obj)
            else:
                return obj
        
        # Convert the modified dict back to a JSON string
        return json.dumps(convert_numeric_to_string(data))
    except json.JSONDecodeError:
        logger.error("Failed to parse metadata JSON for model %s", model_name)
        return None 