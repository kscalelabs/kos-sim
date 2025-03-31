from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np

from typing import Dict
import numpy as np

import json
from pathlib import Path
from typing import TypedDict, List, Dict
from kos_sim.types import ActuatorCommand

class BaseActuator:
    def get_ctrl(
        self,
        kp: float,
        kd: float,
        target_command: float,
        current_position: float,
        current_velocity: float,
        max_torque: float | None = None,
    ) -> float:
        raise NotImplementedError("Subclasses must implement get_ctrl.")
    

class FeetechParams(TypedDict):
    sysid: str
    max_torque: float
    armature: float
    frictionloss: float
    damping: float
    vin: float
    kt: float
    R: float
    error_gain_data: List[Dict[str, float]]

_feetech_config_cache: Dict[str, FeetechParams] = {}

def load_feetech_config_from_catalog(actuator_type: str, base_path: Path) -> FeetechParams:
    catalog_path = base_path / "catalog.json"
    with open(catalog_path, "r") as f:
        catalog = json.load(f)

    actuator_config_relpath = catalog["actuators"].get(actuator_type)
    if actuator_config_relpath is None:
        raise ValueError(f"No config path found for actuator type '{actuator_type}' in catalog.json")

    if actuator_type in _feetech_config_cache:
        return _feetech_config_cache[actuator_type]

    config_path = base_path / actuator_config_relpath
    with open(config_path, "r") as f:
        data = json.load(f)
    _feetech_config_cache[actuator_type] = data
    return data

class FeetechActuator(BaseActuator):
    def __init__(self, actuator_type: str, model_dir: Path):
        self.params = load_feetech_config_from_catalog(actuator_type, model_dir)
        self.max_torque = self.params["max_torque"]

        pos_errs = [d["pos_err"] for d in self.params["error_gain_data"]]
        gains = [d["error_gain"] for d in self.params["error_gain_data"]]
        self._pos_err_min = min(pos_errs)
        self._pos_err_max = max(pos_errs)
        self._error_gain_spline = CubicSpline(pos_errs, gains, extrapolate=True)

    def error_gain(self, error: float) -> float:
        abs_error = abs(error)
        clamped_error = np.clip(abs_error, self._pos_err_min, self._pos_err_max)
        return self._error_gain_spline(clamped_error)

    def get_ctrl(
        self,
        kp: float,
        kd: float,
        target_command: ActuatorCommand,
        current_position: float,
        current_velocity: float,
        max_torque: float | None = None,
    ) -> float:
        pos_error = target_command.get("position", 0.0) - current_position
        vel_error = target_command.get("velocity", 0.0) - current_velocity
        dutycycle = (
            kp * self.error_gain(pos_error) * pos_error +
            kd * vel_error
        )
        torque = np.clip(dutycycle * self.max_torque, -self.max_torque, self.max_torque)
        return torque


class RobstrideActuator(BaseActuator):
    def get_ctrl(
        self,
        kp: float,
        kd: float,
        target_command: ActuatorCommand,
        current_position: float,
        current_velocity: float,
        max_torque: float | None = None,
    ) -> float:
        # Implement Robstride-specific control logic (PD control for now)
        target_torque = (
            kp * (target_command.get("position", 0.0) - current_position)
            + kd * (target_command.get("velocity", 0.0) - current_velocity)
            + target_command.get("torque", 0.0)
        )
        if max_torque is not None:
            target_torque = np.clip(target_torque, -max_torque, max_torque)
        return target_torque


def create_actuator(actuator_type: str, model_dir: Path) -> BaseActuator:
    actuator_type = actuator_type.lower()
    
    if actuator_type.startswith("robstride"):
        return RobstrideActuator()
    elif actuator_type.startswith("feetech"):
        return FeetechActuator(actuator_type, model_dir)
    else:
        raise ValueError(f"Unsupported actuator type: {actuator_type}")


