"""Wrapper around MuJoCo simulation."""

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, TypeVar

import kmv.utils.markers
import kmv.viewer
import mujoco
import numpy as np
from kscale.web.gen.api import RobotURDFMetadataOutput
from mujoco_scenes.mjcf import load_mjmodel

from kos_sim import logger

T = TypeVar("T")

GEOM_TO_MARKER_MAPPING: dict[int, str] = {
    mujoco.mjtGeom.mjGEOM_SPHERE: "SPHERE",
    mujoco.mjtGeom.mjGEOM_BOX: "BOX",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "CAPSULE",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "CYLINDER",
    mujoco.mjtGeom.mjGEOM_ARROW: "ARROW",
}

MARKER_TO_GEOM_MAPPING: dict[str, int] = {v: k for k, v in GEOM_TO_MARKER_MAPPING.items()}


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


class ActuatorCommand(TypedDict):
    position: NotRequired[float]
    velocity: NotRequired[float]
    torque: NotRequired[float]


def get_integrator(integrator: str) -> mujoco.mjtIntegrator:
    match integrator.lower():
        case "euler":
            return mujoco.mjtIntegrator.mjINT_EULER
        case "implicit":
            return mujoco.mjtIntegrator.mjINT_IMPLICIT
        case "implicitfast":
            return mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        case "rk4":
            return mujoco.mjtIntegrator.mjINT_RK4
        case _:
            raise ValueError(f"Invalid integrator: {integrator}")


class MujocoSimulator:
    def __init__(
        self,
        model_path: str | Path,
        model_metadata: RobotURDFMetadataOutput,
        dt: float = 0.001,
        gravity: bool = True,
        render_mode: Literal["window", "offscreen"] = "window",
        freejoint: bool = False,
        start_height: float = 1.5,
        command_delay_min: float = 0.0,
        command_delay_max: float = 0.0,
        joint_pos_delta_noise: float = 0.0,
        joint_pos_noise: float = 0.0,
        joint_vel_noise: float = 0.0,
        pd_update_frequency: float = 100.0,
        mujoco_scene: str = "smooth",
        integrator: str = "implicitfast",
        camera: str | None = None,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> None:
        # Stores parameters.
        self._model_path = model_path
        self._metadata = model_metadata
        self._dt = dt
        self._gravity = gravity
        self._render_mode = render_mode
        self._freejoint = freejoint
        self._start_height = start_height
        self._command_delay_min = command_delay_min
        self._command_delay_max = command_delay_max
        self._joint_pos_delta_noise = math.radians(joint_pos_delta_noise)
        self._joint_pos_noise = math.radians(joint_pos_noise)
        self._joint_vel_noise = math.radians(joint_vel_noise)
        self._update_pd_delta = 1.0 / pd_update_frequency
        self._camera = camera
        self._markers: dict[str, kmv.utils.markers.TrackingMarker] = {}

        # Gets the sim decimation.
        if (control_frequency := self._metadata.control_frequency) is None:
            raise ValueError("Control frequency is not set")
        self._control_frequency = float(control_frequency)
        self._control_dt = 1.0 / self._control_frequency
        self._sim_decimation = int(self._control_dt / self._dt)

        # Gets the joint name mapping.
        if self._metadata.joint_name_to_metadata is None:
            raise ValueError("Joint name to metadata is not set")

        # Gets the IDs, KPs, and KDs for each joint.
        self._joint_name_to_id = {name: _nn(joint.id) for name, joint in self._metadata.joint_name_to_metadata.items()}
        self._joint_name_to_kp: dict[str, float] = {
            name: float(_nn(joint.kp)) for name, joint in self._metadata.joint_name_to_metadata.items()
        }
        self._joint_name_to_kd: dict[str, float] = {
            name: float(_nn(joint.kd)) for name, joint in self._metadata.joint_name_to_metadata.items()
        }
        self._joint_name_to_max_torque: dict[str, float] = {}

        # Gets the inverse mapping.
        self._joint_id_to_name = {v: k for k, v in self._joint_name_to_id.items()}
        if len(self._joint_name_to_id) != len(self._joint_id_to_name):
            raise ValueError("Joint IDs are not unique!")

        # Chooses some random deltas for the joint positions.
        self._joint_name_to_pos_delta = {
            name: random.uniform(-self._joint_pos_delta_noise, self._joint_pos_delta_noise)
            for name in self._joint_name_to_id
        }

        # Load MuJoCo model and initialize data
        logger.info("Loading model from %s", model_path)

        self._model = load_mjmodel(model_path, mujoco_scene)

        self._model.opt.timestep = self._dt
        self._model.opt.integrator = get_integrator(integrator)
        self._model.opt.solver = mujoco.mjtSolver.mjSOL_CG

        self._data = mujoco.MjData(self._model)

        logger.info("Joint ID to name: %s", self._joint_id_to_name)

        if not self._gravity:
            self._model.opt.gravity[2] = 0.0

        # Initialize velocities and accelerations to zero
        if self._freejoint:
            self._data.qpos[:3] = np.array([0.0, 0.0, self._start_height])
            self._data.qpos[3:7] = np.array([0.0, 0.0, 0.0, 1.0])
            self._data.qpos[7:] = np.zeros_like(self._data.qpos[7:])
        else:
            self._data.qpos[:] = np.zeros_like(self._data.qpos)
        self._data.qvel = np.zeros_like(self._data.qvel)
        self._data.qacc = np.zeros_like(self._data.qacc)

        # Important: Step simulation once to initialize internal structures
        mujoco.mj_forward(self._model, self._data)
        mujoco.mj_step(self._model, self._data)

        # Setup viewer after initial step
        self._render_enabled = self._render_mode == "window"

        self._viewer = None

        if self._render_enabled:
            self._viewer = kmv.viewer.MujocoViewerHandler(
                handle=kmv.viewer.launch_passive(
                    self._model,
                    self._data,
                    render_width=frame_width,
                    render_height=frame_height,
                    ctrl_dt=self._dt,
                )
            )

        if self._camera is not None and self._viewer is not None:
            camera_obj = self._model.camera(self._camera)
            self._viewer.setup_camera(
                render_track_body_id=camera_obj.id,
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
        self._actuator_id_to_joint_id = {
            actuator_id: joint_id for joint_id, actuator_id in self._joint_id_to_actuator_id.items()
        }

        # Add control parameters
        self._sim_time = time.time()
        self._current_commands: dict[str, ActuatorCommand] = {
            name: {"position": 0.0, "velocity": 0.0, "torque": 0.0} for name in self._joint_name_to_id
        }
        self._next_commands: dict[str, tuple[ActuatorCommand, float]] = {}

    async def step(self) -> None:
        """Execute one step of the simulation."""
        self._sim_time += self._dt

        # Process commands that are ready to be applied
        commands_to_remove = []
        for name, (target_command, application_time) in self._next_commands.items():
            if self._sim_time >= application_time:
                self._current_commands[name] = target_command
                commands_to_remove.append(name)

        # Remove processed commands
        if commands_to_remove:
            for name in commands_to_remove:
                self._next_commands.pop(name)

        mujoco.mj_forward(self._model, self._data)

        # Sets the ctrl values from the current commands.
        for name, target_command in self._current_commands.items():
            joint_id = self._joint_name_to_id[name]
            actuator_id = self._joint_id_to_actuator_id[joint_id]
            kp = self._joint_name_to_kp[name]
            kd = self._joint_name_to_kd[name]
            current_position = self._data.joint(name).qpos
            current_velocity = self._data.joint(name).qvel
            target_torque = (
                kp * (target_command["position"] - current_position)
                + kd * (target_command["velocity"] - current_velocity)
                + target_command["torque"]
            )
            if (max_torque := self._joint_name_to_max_torque.get(name)) is not None:
                target_torque = np.clip(target_torque, -max_torque, max_torque)
            logger.debug("Setting ctrl for actuator %s to %f", actuator_id, target_torque)
            self._data.ctrl[actuator_id] = target_torque

        # Step physics - allow other coroutines to run during computation

        # for some reason running forward before step makes it more stable.
        # It possibly computes some values that are needed for the step.
        mujoco.mj_forward(self._model, self._data)
        mujoco.mj_step(self._model, self._data)

        return self._data

    async def render(self) -> None:
        """Render the simulation asynchronously."""
        if self._render_enabled:
            assert self._viewer is not None
            for marker in self._markers.values():
                self._viewer._markers.append(marker)

            self._viewer.update_and_sync()

    async def capture_frame(self, camid: int = -1, depth: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        """Capture a frame from the simulation using read_pixels.

        Args:
            camid: Camera ID to use (-1 for free camera)
            depth: Whether to return depth information

        Returns:
            RGB image array (and optionally depth array) if depth=True
        """
        # TODO: Use native mujoco renderder for offline and shiiiiiiiiit
        return np.zeros((480, 640, 3), dtype=np.uint8), None
        # if self._render_mode != "offscreen" and self._render_enabled:
        #     logger.warning("Capturing frames is more efficient in offscreen mode")

        # if depth:
        #     logger.warning("Depth is not currently supported")

        # for marker in self._markers.values():
        #     self._viewer.add_marker(**marker)

        # if camid is not None:
        #     if camid == -1:
        #         self._viewer.handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        #     else:
        #         self._viewer.handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        #         self._viewer.handle.cam.fixedcamid = camid

        # rgb = self._viewer.read_pixels()
        # return rgb, None

    async def get_sensor_data(self, name: str) -> np.ndarray:
        """Get data from a named sensor."""
        if name not in self._sensor_name_to_id:
            raise KeyError(f"Sensor '{name}' not found")
        sensor_id = self._sensor_name_to_id[name]
        return self._data.sensor(sensor_id).data.copy()

    async def get_actuator_state(self, joint_id: int) -> ActuatorState:
        """Get current state of an actuator using real joint ID."""
        if joint_id not in self._joint_id_to_name:
            raise KeyError(f"Joint ID {joint_id} not found in config mappings")

        joint_name = self._joint_id_to_name[joint_id]
        joint_data = self._data.joint(joint_name)

        return ActuatorState(
            position=float(joint_data.qpos)
            + self._joint_name_to_pos_delta[joint_name]
            + random.uniform(-self._joint_pos_noise, self._joint_pos_noise),
            velocity=float(joint_data.qvel) + random.uniform(-self._joint_vel_noise, self._joint_vel_noise),
        )

    async def command_actuators(self, commands: dict[int, ActuatorCommand]) -> None:
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

            self._next_commands[joint_name] = (command, application_time)

    async def configure_actuator(self, joint_id: int, configuration: ConfigureActuatorRequest) -> None:
        """Configure an actuator using real joint ID."""
        if joint_id not in self._joint_id_to_actuator_id:
            raise KeyError(
                f"Joint ID {joint_id} not found in config mappings. "
                f"The available joint IDs are {self._joint_id_to_actuator_id.keys()}"
            )

        joint_name = self._joint_id_to_name[joint_id]
        if "kp" in configuration:
            self._joint_name_to_kp[joint_name] = configuration["kp"]
        if "kd" in configuration:
            self._joint_name_to_kd[joint_name] = configuration["kd"]
        if "max_torque" in configuration:
            self._joint_name_to_max_torque[joint_name] = configuration["max_torque"]

    @property
    def sim_time(self) -> float:
        return self._sim_time

    async def reset(
        self,
        xyz: tuple[float, float, float] | None = None,
        quat: tuple[float, float, float, float] | None = None,
        joint_pos: dict[str, float] | None = None,
        joint_vel: dict[str, float] | None = None,
    ) -> None:
        """Reset simulation to specified or default state."""
        self._next_commands.clear()

        mujoco.mj_resetData(self._model, self._data)

        # Resets qpos.
        qpos = np.zeros_like(self._data.qpos)
        if self._freejoint:
            qpos[:3] = np.array([0.0, 0.0, self._start_height] if xyz is None else xyz)
            qpos[3:7] = np.array([0.0, 0.0, 0.0, 1.0] if quat is None else quat)
            qpos[7:] = np.zeros_like(self._data.qpos[7:])
        else:
            qpos[:] = np.zeros_like(self._data.qpos)

        if joint_pos is not None:
            for joint_name, position in joint_pos.items():
                self._data.joint(joint_name).qpos = position

        # Resets qvel.
        qvel = np.zeros_like(self._data.qvel)
        if joint_vel is not None:
            for joint_name, velocity in joint_vel.items():
                self._data.joint(joint_name).qvel = velocity

        # Resets qacc.
        qacc = np.zeros_like(self._data.qacc)

        # Runs one step.
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._data.qacc[:] = qacc
        mujoco.mj_forward(self._model, self._data)
        self._current_commands.clear()

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
