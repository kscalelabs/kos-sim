from scipy.optimize import curve_fit
import pandas as pd
import numpy as np

from typing import Dict
import numpy as np

import json
from pathlib import Path
from typing import TypedDict, List, Dict
from kos_sim.types import ActuatorCommand
from kos_sim import logger



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

        # Default a/x + b parameters
        self.a_param = 0.00005502  # Default value (STS3250)
        self.b_param = 0.16293639  # Default value (STS3250)
        self._pos_err_min = 0.001
        self._pos_err_max = 0.15

        # Extract error gain data and fit a/x + b curve
        error_data = self.params["error_gain_data"]
        if error_data and len(error_data) <= 3:
            logger.warning(f"Not enough error gain data for {actuator_type}. Using default values.")
        else:
            try:
                # Sort by position error
                error_data_sorted = sorted(error_data, key=lambda d: d["pos_err"])
                pos_errs = np.array([d["pos_err"] for d in error_data_sorted])
                gains = np.array([d["error_gain"] for d in error_data_sorted])
                
                # Store min/max position errors for clamping
                self._pos_err_min = float(np.min(pos_errs))
                self._pos_err_max = float(np.max(pos_errs))

                def inverse_func(x, a, b):
                    return a/x + b
                popt, _ = curve_fit(inverse_func, pos_errs, gains)
                self.a_param, self.b_param = popt
                logger.info(
                    f"Actuator {actuator_type}: Fitted parameters a={self.a_param:.6f}, b={self.b_param:.6f}, "
                    f"error range [{self._pos_err_min:.6f}, {self._pos_err_max:.6f}]"
                )
            except Exception as e:
                logger.warning(f"Error fitting curve for {actuator_type}: {e}. Using default values.")
                # Use defaults if fitting fails
                self._pos_err_min = 0.001
                self._pos_err_max = 0.15

        logger.debug(
            f"Initializing FeetechActuator with params: "
            f"max_torque={self.max_torque}, "
            f"max_pwm={self.params.get('max_pwm', 1.0)}, "
            f"vin={self.params.get('vin', 12.0)}, "
            f"a={self.a_param:.6f}, b={self.b_param:.6f}"
        )

    def error_gain(self, error: float) -> float:
        abs_error = abs(error)
        # Clamp to the min/max range from error data (matching firmware behavior)
        clamped_error = np.clip(abs_error, self._pos_err_min, self._pos_err_max)
        # Use reciprocal function a/x + b
        return self.a_param / clamped_error + self.b_param

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

        if self.prev_target_position is None:
            self.prev_target_position = current_position

        # Differentiate target position to get velocity
        expected_velocity = (target_position - self.prev_target_position) / dt
        self.prev_target_position = target_position 
            
        # Calculate errors
        pos_error = target_position - current_position
        vel_error = expected_velocity - current_velocity

        # Calculate duty cycle with error gain scaling
        error_gain = self.error_gain(pos_error)
        raw_duty = (
            kp * error_gain * pos_error +
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


