"""Utility functions and classes."""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class JointMetadataOutput(BaseModel):
    """Metadata for a joint in the robot URDF."""

    id: Optional[int] = Field(None, title="Id")
    kp: Optional[str] = Field(None, title="Kp")
    kd: Optional[str] = Field(None, title="Kd")
    armature: Optional[str] = Field(None, title="Armature")
    friction: Optional[str] = Field(None, title="Friction")
    offset: Optional[str] = Field(None, title="Offset")
    flipped: Optional[bool] = Field(None, title="Flipped")


class RobotURDFMetadataOutput(BaseModel):
    """Metadata for the robot URDF."""

    joint_name_to_metadata: Optional[Dict[str, JointMetadataOutput]] = Field(None, title="Joint Name To Metadata")
    control_frequency: Optional[str] = Field(None, title="Control Frequency")
