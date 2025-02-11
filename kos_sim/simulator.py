"""Wrapper around MuJoCo simulation."""

from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict, TypeVar

import mujoco
import mujoco_viewer
import numpy as np
from kscale.web.gen.api import RobotURDFMetadataOutput

from kos_sim import logger

T = TypeVar("T")


def _nn(value: T | None) -> T:
    if value is None:
        raise ValueError("Value is not set")
    return value


class ConfigureActuatorRequest(TypedDict):
    torque_enabled: NotRequired[bool]
    zero_position: NotRequired[float]
    kp: NotRequired[float]
    kd: NotRequired[float]
    max_torque: NotRequired[float]


@dataclass
class ActuatorState:
    position: float
    velocity: float
    effort: float | None = None


class MujocoSimulator:
    def __init__(
        self,
        model_path: str | Path,
        model_metadata: RobotURDFMetadataOutput,
        dt: float = 0.001,
        gravity: bool = True,
        render: bool = True,
        suspended: bool = False,
        command_delay_min: float = 0.0,
        command_delay_max: float = 0.0,
    ) -> None:
        # Stores parameters.
        self._model_path = model_path
        self._metadata = model_metadata
        self._dt = dt
        self._gravity = gravity
        self._render = render
        self._suspended = suspended
        self._command_delay_min = command_delay_min
        self._command_delay_max = command_delay_max

        # Gets the sim decimation.
        if (control_frequency := self._metadata.control_frequency) is None:
            raise ValueError("Control frequency is not set")
        self._control_frequency = float(control_frequency)
        self._sim_decimation = int(1 / self._control_frequency / self._dt)

        # Gets the joint name mapping.
        if self._metadata.joint_name_to_metadata is None:
            raise ValueError("Joint name to metadata is not set")
        self._joint_name_to_id = {name: _nn(joint.id) for name, joint in self._metadata.joint_name_to_metadata.items()}
        self._joint_id_to_name = {v: k for k, v in self._joint_name_to_id.items()}
        if len(self._joint_name_to_id) != len(self._joint_id_to_name):
            raise ValueError("Joint IDs are not unique!")

        logger.info("Joint ID to name: %s", self._joint_id_to_name)

        # Load MuJoCo model and initialize data
        logger.info("Loading model from %s", model_path)
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._model.opt.timestep = self._dt
        self._data = mujoco.MjData(self._model)

        self._gravity = self._gravity
        self._suspended = self._suspended
        self._initial_pos = None
        self._initial_quat = None

        if not self._gravity:
            self._model.opt.gravity[2] = 0.0

        # If suspended, store initial position and orientation
        if self._suspended:
            # Find the free joint that controls base position and orientation
            for i in range(self._model.njnt):
                if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    self._initial_pos = self._data.qpos[i : i + 3].copy()  # xyz position
                    self._initial_quat = self._data.qpos[i + 3 : i + 7].copy()  # quaternion
                    break

        # Initialize velocities and accelerations to zero
        self._data.qvel = np.zeros_like(self._data.qvel)
        self._data.qacc = np.zeros_like(self._data.qacc)

        # Important: Step simulation once to initialize internal structures
        mujoco.mj_step(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

        # Setup viewer after initial step
        self._render_enabled = self._render
        self._viewer = mujoco_viewer.MujocoViewer(
            self._model,
            self._data,
            mode="window" if self._render_enabled else "offscreen",
        )

        # Cache lookups after initialization
        self._sensor_name_to_id = {self._model.sensor(i).name: i for i in range(self._model.nsensor)}
        logger.debug("Sensor IDs: %s", self._sensor_name_to_id)
        self._actuator_name_to_id = {self._model.actuator(i).name: i for i in range(self._model.nu)}
        logger.debug("Actuator IDs: %s", self._actuator_name_to_id)

        # There is an important distinction between actuator IDs and joint IDs.
        # joint IDs should be at the kos layer, where the canonical IDs are assigned (see docs.kscale.dev)
        # but actuator IDs are at the mujoco layer, where the actuators actually get mapped.
        logger.debug("Joint ID to name: %s", self._joint_id_to_name)
        self._joint_id_to_actuator_id = {
            joint_id: self._actuator_name_to_id[f"{name}_ctrl"] for joint_id, name in self._joint_id_to_name.items()
        }

        # Add control parameters
        self._sim_time = 0.0
        self._current_commands: dict[str, tuple[float, float]] = {}

    async def step(self) -> None:
        """Execute one step of the simulation."""
        self._sim_time += self._dt

        # Process commands that are ready to be applied
        commands_to_remove = []
        for name, (target_pos, application_time) in self._current_commands.items():
            if self._sim_time >= application_time:
                actuator_id = self._actuator_name_to_id[name]
                logger.debug("Setting actuator %s (id %d) to %f", name, actuator_id, target_pos)
                self._data.ctrl[actuator_id] = target_pos
                commands_to_remove.append(name)

        # Remove processed commands
        if commands_to_remove:
            logger.debug("Removing %d commands at sim time %f", len(commands_to_remove), self._sim_time)
            for name in commands_to_remove:
                self._current_commands.pop(name)

        # Step physics - allow other coroutines to run during computation
        mujoco.mj_step(self._model, self._data)
        if self._suspended:
            # Find the root joint (floating_base)
            for i in range(self._model.njnt):
                if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    self._data.qpos[i : i + 7] = self._model.keyframe("default").qpos[i : i + 7]
                    self._data.qvel[i : i + 6] = 0
                    break

        return self._data

    async def render(self) -> None:
        """Render the simulation asynchronously."""
        if self._render_enabled:
            self._viewer.render()

    async def get_sensor_data(self, name: str) -> np.ndarray:
        """Get data from a named sensor."""
        if name not in self._sensor_name_to_id:
            raise KeyError(f"Sensor '{name}' not found")
        sensor_id = self._sensor_name_to_id[name]
        return self._data.sensor(sensor_id).data.copy()

    async def get_actuator_state(self, joint_id: int) -> ActuatorState:
        """Get current state of an actuator using real joint ID."""
        logger.debug("Getting actuator state for joint ID: %s", joint_id)
        if joint_id not in self._joint_id_to_name:
            raise KeyError(f"Joint ID {joint_id} not found in config mappings")

        joint_name = self._joint_id_to_name[joint_id]
        joint_data = self._data.joint(joint_name)

        return ActuatorState(
            position=float(joint_data.qpos),
            velocity=float(joint_data.qvel),
            # effort=float(joint_data.qfrc_ext),
        )

    async def command_actuators(self, commands: dict[int, float]) -> None:
        """Command multiple actuators at once using real joint IDs."""
        for joint_id, command in commands.items():
            # Translate real joint ID to MuJoCo joint name
            if joint_id not in self._joint_id_to_name:
                logger.warning("Joint ID %d not found in config mappings", joint_id)
                continue

            joint_name = self._joint_id_to_name[joint_id]
            actuator_name = f"{joint_name}_ctrl"
            if actuator_name not in self._actuator_name_to_id:
                logger.warning("Joint %s not found in MuJoCo model", actuator_name)
                continue

            # Calculate random delay and application time
            delay = np.random.uniform(self._command_delay_min, self._command_delay_max)
            application_time = self._sim_time + delay

            logger.debug("Setting actuator %s to %f with delay %f s", actuator_name, command, delay)
            self._current_commands[actuator_name] = (command, application_time)

    async def configure_actuator(self, joint_id: int, configuration: ConfigureActuatorRequest) -> None:
        """Configure an actuator using real joint ID."""
        if joint_id not in self._joint_id_to_actuator_id:
            raise KeyError(
                f"Joint ID {joint_id} not found in config mappings. "
                f"The available joint IDs are {self._joint_id_to_actuator_id.keys()}"
            )
        actuator_id = self._joint_id_to_actuator_id[joint_id]

        if "kp" in configuration:
            prev_kp = float(self._model.actuator_gainprm[actuator_id, 0])
            self._model.actuator_gainprm[actuator_id, 0] = configuration["kp"]
            logger.debug("Set kp for actuator %s from %f to %f", joint_id, prev_kp, configuration["kp"])

        if "kd" in configuration:
            prev_kd = -float(self._model.actuator_biasprm[actuator_id, 2])
            self._model.actuator_biasprm[actuator_id, 2] = -configuration["kd"]
            logger.debug("Set kd for actuator %s from %f to %f", joint_id, prev_kd, configuration["kd"])

        if "max_torque" in configuration:
            prev_min_torque = float(self._model.actuator_forcerange[actuator_id, 0])
            prev_max_torque = float(self._model.actuator_forcerange[actuator_id, 1])
            self._model.actuator_forcerange[actuator_id, 0] = -configuration["max_torque"]
            self._model.actuator_forcerange[actuator_id, 1] = configuration["max_torque"]
            logger.debug(
                "Set max_torque for actuator %s from (%f, %f) to +/- %f",
                joint_id,
                prev_min_torque,
                prev_max_torque,
                configuration["max_torque"],
            )

    async def reset(self, qpos: list[float] | None = None) -> None:
        """Reset simulation to specified or default state."""
        self._sim_time = 0.0
        self._current_commands.clear()

        mujoco.mj_resetData(self._model, self._data)
        if qpos is not None:
            self._data.qpos[: len(qpos)] = qpos
        self._data.qvel[:] = np.zeros_like(self._data.qvel)
        self._data.qacc[:] = np.zeros_like(self._data.qacc)
        mujoco.mj_forward(self._model, self._data)

    async def close(self) -> None:
        """Clean up simulation resources."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception as e:
                logger.error("Error closing viewer: %s", e)
            self._viewer = None

    @property
    def timestep(self) -> float:
        return self._model.opt.timestep
