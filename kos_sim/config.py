"""Configuration for the simulator."""

import logging
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    joint_id_to_name: dict[int, str]
    joint_name_to_id: dict[str, int]
    kp: float = 32.0
    kd: float = 10.0

    @classmethod
    def from_file(cls, config_path: str) -> "SimulatorConfig":
        """Load config from YAML file."""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Load joint mappings
        joint_name_to_id = config_data.get("joint_mappings", {})
        joint_id_to_name = {v: k for k, v in joint_name_to_id.items()}

        # Load control parameters
        kp = config_data.get("control", {}).get("kp", 1.0)
        kd = config_data.get("control", {}).get("kd", 0.1)

        return cls(joint_id_to_name=joint_id_to_name, joint_name_to_id=joint_name_to_id, kp=kp, kd=kd)

    @classmethod
    def default(cls) -> "SimulatorConfig":
        """Create default config with standard joint mappings."""
        joint_name_to_id = {
            "L_hip_y": 1,
            "L_hip_x": 2,
            "L_hip_z": 3,
            "L_knee": 4,
            "L_ankle_y": 5,
            "R_hip_y": 6,
            "R_hip_x": 7,
            "R_hip_z": 8,
            "R_knee": 9,
            "R_ankle_y": 10,
        }
        return cls(joint_id_to_name={v: k for k, v in joint_name_to_id.items()}, joint_name_to_id=joint_name_to_id)
