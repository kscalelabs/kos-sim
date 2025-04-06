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
        dt: float | None = None,
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
        
        # Store additional parameters
        self.max_velocity = self.params.get("max_velocity", 10.0)  # Default if not specified
        self.max_pwm = self.params.get("max_pwm", 1.0)  # Default max duty cycle if not specified 
        self.vin = self.params.get("vin", 12.0)  # Default input voltage
        self.kt = self.params.get("kt", 0.18)  # Default torque constant
        self.R = self.params.get("R", 1.0)  # Default resistance
        
        # For velocity smoothing
        self.dt = None  # Default, will be overridden if set
        self.prev_target_position = None

        # Error gain spline
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
        dt: float | None = None,
    ) -> float:
        # Use instance max_torque if none provided
        if max_torque is None:
            max_torque = self.max_torque
            
        # Get target position from command
        target_position = target_command.get("position", current_position)
        #velocity_limit = target_command.get("velocity", 0.0) # Not currently used
        
        if self.prev_target_position is None:
            self.prev_target_position = current_position

        # Differentiate target position to get velocity
        expected_velocity = (target_position - self.prev_target_position) / dt
        self.prev_target_position = target_position  # Update for next time
            
        # Calculate errors
        pos_error = target_position - current_position
        vel_error = expected_velocity - current_velocity

        # Calculate duty cycle with error gain scaling
        raw_duty = (
            kp * self.error_gain(pos_error) * pos_error +
            kd * vel_error
        )

        # Clip duty cycle based on max_pwm
        duty = np.clip(raw_duty, -self.max_pwm, self.max_pwm)
        
        # Calculate voltage and torque using motor electrical model
        voltage = duty * self.vin
        torque = voltage * self.kt / self.R
        
        # Clip to max torque
        torque = np.clip(torque, -max_torque, max_torque)
        
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
        dt: float | None = None,
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


