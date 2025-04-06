from typing import TypedDict, NotRequired

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


