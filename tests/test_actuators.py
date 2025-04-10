"""Tests for actuator models in KOS-Sim."""

from pathlib import Path

import pytest

from kos_sim.actuators import FeetechActuator
from kos_sim.types import ActuatorCommand

TEST_ASSETS_DIR = Path(__file__).parent / "assets"
MODEL_DIR = TEST_ASSETS_DIR / "actuators"
CATALOG_PATH = MODEL_DIR / "catalog.json"

import json

# Load the catalog to determine which parameter files and actuator types to test
with open(CATALOG_PATH, "r") as catalog_file:
    catalog = json.load(catalog_file)

ACTUATOR_PARAMS_FILES = [MODEL_DIR / rel_path for rel_path in catalog["actuators"].values()]
ACTUATOR_TYPES = list(catalog["actuators"].keys())

assert len(ACTUATOR_PARAMS_FILES) == len(ACTUATOR_TYPES)

# Fail early if any parameter file listed in the catalog is missing
for params_path in ACTUATOR_PARAMS_FILES:
    if not params_path.is_file():
        raise FileNotFoundError(f"Required actuator params file not found during test setup: {params_path}")


@pytest.mark.parametrize("actuator_type", ACTUATOR_TYPES)
def test_feetech_actuator_basic(actuator_type):  # Removed check_asset_files dependency
    """Tests basic initialization and get_ctrl call for FeetechActuator using real params."""
    print(f"\n--- Testing actuator type: {actuator_type} ---")
    model_dir = MODEL_DIR
    feetech_actuator = FeetechActuator(actuator_type, model_dir)

    assert hasattr(feetech_actuator, "max_torque")
    assert isinstance(feetech_actuator.max_torque, float)

    kp = 20.0
    kd = 5.0
    target_command: ActuatorCommand = {"position": 1.0}  # Target position 1.0 rad
    current_position = 0.0
    current_velocity = 0.0
    dt = 0.001  # Needs a dt for velocity calculation

    print(f"kp: {kp}, kd: {kd}")

    print(f"Sending target command: {target_command}")

    torque = feetech_actuator.get_ctrl(
        kp=kp,
        kd=kd,
        target_command=target_command,
        current_position=current_position,
        current_velocity=current_velocity,
        dt=dt,
    )

    print(f"Generated torque: {torque}")

    assert isinstance(torque, float)
    assert torque >= 0
    assert abs(torque) <= feetech_actuator.max_torque
