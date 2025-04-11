"""Actuator models for KOS-Sim."""

import json
from pathlib import Path
import numpy as np
from kos_sim import logger
from kos_sim.types import ActuatorCommand, FeetechParams


class BaseActuator:
    @property
    def is_stateful(self) -> bool:
        """Return whether this actuator maintains state between calls."""
        return False

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
        raise NotImplementedError("Subclasses must implement get_ctrl.")
        

class RobstrideActuator(BaseActuator):
    # Since RobstrideActuator is stateless, we can make it a singleton
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RobstrideActuator, cls).__new__(cls)
        return cls._instance

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


class FeetechActuator(BaseActuator):
    @property
    def is_stateful(self) -> bool:
        return True
    
    def __init__(self, actuator_type: str, params_path: Path) -> None:
        self.params = load_feetech_params(actuator_type, params_path)
        self._validate_params()

        self.max_torque = self.params["max_torque"]
        self.max_velocity = self.params["max_velocity"]
        self.max_pwm = self.params["max_pwm"]
        self.vin = self.params["vin"]
        self.kt = self.params["kt"]
        self.R = self.params["R"]
        self.error_gain = self.params["error_gain"]
        self.dt = None  # Default, will be overridden if set
        self.vmax = 5.0
        self.acceleration= 39.0
        self.motion_planner = TrapezoidalPlanner(self.vmax, self.acceleration)

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
        self.motion_planner.set_target(target_position)

        desired_position, desired_velocity = self.motion_planner.update(dt)
        
        # Calculate errors
        pos_error = desired_position - current_position
        vel_error = desired_velocity - current_velocity

        # Calculate duty cycle
        raw_duty = kp * self.error_gain * pos_error + kd * vel_error

        # Clip duty cycle based on max_pwm
        duty = np.clip(raw_duty, -self.max_pwm, self.max_pwm)

        # Calculate voltage and torque using motor electrical model
        voltage = duty * self.vin
        torque = voltage * self.kt / self.R

        # Clip to max torque
        torque = np.clip(torque, -max_torque, max_torque)
        #print(f"torque: {torque}, max_torque: {max_torque}, duty: {duty}, voltage: {voltage}, torque: {torque}")

        return torque

def create_actuator(actuator_type: str, params_path: Path) -> BaseActuator:
    actuator_type = actuator_type.lower()

    if actuator_type.startswith("robstride"):
        return RobstrideActuator()  # Singleton
    elif actuator_type.startswith("feetech"):
        return FeetechActuator(actuator_type, params_path)  # Stateful
    else:
        raise ValueError(f"Unsupported actuator type: {actuator_type}")

def load_feetech_params(actuator_type: str, params_path: Path) -> FeetechParams:
    """Load actuator parameters directly from a JSON file."""
    config_path = params_path / f"{actuator_type}.json"
    if not config_path.exists():
        raise ValueError(f"Actuator parameters file '{config_path}' not found")
    
    with open(config_path, "r") as f:
        return json.load(f)

class TrapezoidalPlanner:
    def __init__(self, v_max: float, acceleration: float):
        self.v_max = v_max  # maximum velocity
        self.acceleration = acceleration  # constant acceleration
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.target_position = 0.0

    def set_target(self, target_position: float):
        self.target_position = target_position

    def update(self, dt: float):
        position_error = self.target_position - self.current_position
        direction = np.sign(position_error)

        # Distance needed to stop
        stopping_distance = (self.current_velocity ** 2) / (2 * self.acceleration)

        # Decision: Accelerate, cruise, or decelerate
        if abs(position_error) > stopping_distance:
            # Accelerate towards target
            self.current_velocity += direction * self.acceleration * dt
            # Limit to max velocity
            self.current_velocity = np.clip(self.current_velocity, -self.v_max, self.v_max)
        else:
            # Decelerate to stop at target
            self.current_velocity -= direction * self.acceleration * dt
            # Clamp velocity to avoid overshoot
            if direction * self.current_velocity < 0:
                self.current_velocity = 0.0

        # Update position
        self.current_position += self.current_velocity * dt

        return self.current_position, self.current_velocity
