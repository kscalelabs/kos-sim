"""Types for the simulator."""

from typing import NotRequired, TypedDict


class ActuatorCommand(TypedDict):
    position: NotRequired[float]
    velocity: NotRequired[float]
    torque: NotRequired[float]


class ConfigureActuatorRequest(TypedDict):
    torque_enabled: NotRequired[bool]
    zero_position: NotRequired[float]
    kp: NotRequired[float]
    kd: NotRequired[float]
    max_torque: NotRequired[float]
    acceleration: NotRequired[float]

class FeetechParams(TypedDict):
    sysid: str
    max_torque: float
    armature: float
    frictionloss: float
    damping: float
    vin: float
    kt: float
    R: float
    error_gain: float
