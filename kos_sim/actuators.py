"""Actuator models for KOS-Sim."""

import json
from pathlib import Path
from typing import Dict, List, TypedDict

import numpy as np

from kos_sim import logger
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
    def __init__(self, actuator_type: str, model_dir: Path) -> None:
        self.params = load_feetech_config_from_catalog(actuator_type, model_dir)
        self._validate_params()

        self.max_torque = self.params["max_torque"]
        self.max_velocity = self.params["max_velocity"]
        self.max_pwm = self.params["max_pwm"]
        self.vin = self.params["vin"]
        self.kt = self.params["kt"]
        self.R = self.params["R"]
        self.error_gain = self.params["error_gain"]
        self.prev_target_position = None
        self.dt = None  # Default, will be overridden if set
        logger.debug(
            f"Initializing FeetechActuator with params: "
            f"max_torque={self.max_torque}, "
            f"max_pwm={self.params.get('max_pwm', 1.0)}, "
            f"vin={self.params.get('vin', 12.0)}, "
            f"kt={self.kt}, "
            f"R={self.R}, "
            f"error_gain={self.error_gain}"
        )

    def _validate_params(self):
        """Validate all required parameters are present with valid values."""
        required_params = {
            "max_torque": (float, "Maximum torque in N⋅m"),
            "error_gain": (float, "Error gain scaling factor"),
            "max_velocity": (float, 10.0, "Maximum velocity in rad/s"),
            "max_pwm": (float, 1.0, "Maximum duty cycle"),
            "vin": (float, 12.0, "Input voltage in V"),
            "kt": (float, 0.18, "Torque constant in N⋅m/A"),
            "R": (float, 1.0, "Motor resistance in Ω"),
        }

        # Check required parameters
        for param, param_type in required_params.items():
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
            if not isinstance(self.params[param], param_type):
                raise TypeError(f"Parameter {param} must be {param_type.__name__}")
            if self.params[param] <= 0:
                raise ValueError(f"Parameter {param} must be positive")

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

        # Calculate duty cycle
        raw_duty = kp * self.error_gain * pos_error + kd * vel_error

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
