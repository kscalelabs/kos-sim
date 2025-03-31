from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np

from typing import Dict
import numpy as np


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

class RobstrideActuator(BaseActuator):
    def get_ctrl(
        self,
        kp: float,
        kd: float,
        target_command: float,
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

class FeetechActuator(BaseActuator):
    def get_ctrl(
        self,
        kp: float,
        kd: float,
        target_command: float,
        current_position: float,
        current_velocity: float,
        max_torque: float | None = None,
    ) -> float:
        # For now, use the same PD control logic.
        target_torque = (
            kp * (target_command.get("position", 0.0) - current_position)
            + kd * (target_command.get("velocity", 0.0) - current_velocity)
            + target_command.get("torque", 0.0)
        )
        if max_torque is not None:
            target_torque = np.clip(target_torque, -max_torque, max_torque)
        return target_torque

def create_actuator(actuator_type: str) -> BaseActuator:
    actuator_type = actuator_type.lower()
    if actuator_type.startswith("robstride"):
        return RobstrideActuator()
    elif actuator_type.startswith("feetech"):
        return FeetechActuator()
    else:
        raise ValueError(f"Unsupported actuator type: {actuator_type}")
