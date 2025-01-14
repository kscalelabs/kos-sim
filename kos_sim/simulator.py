"""Wrapper around MuJoCo simulation."""

import logging
import threading

import mujoco
import mujoco_viewer
import numpy as np

from kos_sim.config import SimulatorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MujocoSimulator:
    def __init__(
        self,
        model_path: str,
        config: SimulatorConfig | None = None,
        render: bool = False,
        dt: float = 0.002,
        gravity: bool = True,
        suspended: bool = False,
    ) -> None:
        # Load config or use default
        self._config = config or SimulatorConfig.default()

        # Load MuJoCo model and initialize data
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._model.opt.timestep = dt
        self._data = mujoco.MjData(self._model)

        self._gravity = gravity
        self._suspended = suspended
        self._initial_pos = None
        self._initial_quat = None

        if not self._gravity:
            self._model.opt.gravity[2] = 0.0

        # Initialize default position from keyframe if available
        try:
            self._data.qpos = self._model.keyframe("default").qpos
            logger.info("Loaded default position from keyframe")
        except Exception:
            logger.warning("No default keyframe found, using zero initialization")
            self._data.qpos = np.zeros_like(self._data.qpos)

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
        self._render_enabled = render
        if render:
            self._viewer = mujoco_viewer.MujocoViewer(self._model, self._data)
        else:
            self._viewer = mujoco_viewer.MujocoViewer(self._model, self._data, "offscreen")

        # Cache lookups after initialization
        self._sensor_ids = {self._model.sensor(i).name: i for i in range(self._model.nsensor)}
        self._actuator_ids = {self._model.actuator(i).name: i for i in range(self._model.nu)}

        # Add control parameters
        self._lock = threading.Lock()
        self._current_commands: dict[str, float] = {}

        # Use config for control parameters
        self._kp = np.array([self._config.kp] * self._model.nu)
        self._kd = np.array([self._config.kd] * self._model.nu)

        self._count_lowlevel = 0
        self._target_positions: dict[str, float] = {}  # Store target positions between updates

    def step(self) -> None:
        """Execute one step of the simulation."""
        with self._lock:
            # Only update commands every sim_decimation steps
            if self._count_lowlevel % self._config.sim_decimation == 0:
                self._target_positions = self._current_commands.copy()

            # Apply actuator commands using PD control
            for name, target_pos in self._target_positions.items():
                actuator_id = self._actuator_ids[name]
                current_pos = self._data.qpos[actuator_id]
                current_vel = self._data.qvel[actuator_id]

                # PD control law
                tau = self._kp[actuator_id] * (target_pos - current_pos) - self._kd[actuator_id] * current_vel
                self._data.ctrl[actuator_id] = tau

        # Step physics
        mujoco.mj_step(self._model, self._data)

        # Increment counter
        self._count_lowlevel += 1

        if self._suspended:
            # Find the root joint (floating_base)
            for i in range(self._model.njnt):
                if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    print(f"Joint name: {self._model.joint(i).name}")
                    self._data.qpos[i:i + 7] = self._model.keyframe("default").qpos[i:i + 7]
                    self._data.qvel[i:i + 6] = 0
                    break

        if self._render_enabled:
            self._viewer.render()

    def get_sensor_data(self, name: str) -> np.ndarray:
        """Get data from a named sensor."""
        if name not in self._sensor_ids:
            raise KeyError(f"Sensor '{name}' not found")
        sensor_id = self._sensor_ids[name]
        return self._data.sensor(sensor_id).data.copy()

    def get_actuator_state(self, joint_id: int) -> float:
        """Get current state of an actuator using real joint ID."""
        if joint_id not in self._config.joint_id_to_name:
            raise KeyError(f"Joint ID {joint_id} not found in config mappings")

        joint_name = self._config.joint_id_to_name[joint_id]
        if joint_name not in self._actuator_ids:
            raise KeyError(f"Joint {joint_name} not found in MuJoCo model")

        actuator_id = self._actuator_ids[joint_name]
        return float(self._data.qpos[actuator_id])

    def command_actuators(self, commands: dict[int, float]) -> None:
        """Command multiple actuators at once using real joint IDs."""
        with self._lock:
            for joint_id, command in commands.items():
                # Translate real joint ID to MuJoCo joint name
                if joint_id not in self._config.joint_id_to_name:
                    logger.warning("Joint ID %d not found in config mappings", joint_id)
                    continue

                joint_name = self._config.joint_id_to_name[joint_id]
                if joint_name not in self._actuator_ids:
                    logger.warning("Joint %s not found in MuJoCo model", joint_name)
                    continue

                self._current_commands[joint_name] = command

    def reset(self, position: dict[str, float] | None = None, orientation: list[float] | None = None) -> None:
        """Reset simulation to specified or default state.

        Args:
            position: Dict of joint names to positions (radians)
            orientation: Quaternion [w, x, y, z] for base orientation
        """
        mujoco.mj_resetData(self._model, self._data)

        # Set joint positions if provided, otherwise use defaults
        if position is not None:
            for joint_name, pos in position.items():
                if joint_name in self._actuator_ids:
                    self._data.qpos[self._actuator_ids[joint_name]] = pos
        else:
            try:
                self._data.qpos = self._model.keyframe("default").qpos
            except Exception:
                self._data.qpos = np.zeros_like(self._data.qpos)

        # Set orientation if provided
        if orientation is not None:
            # Find the free joint that controls base orientation
            for i in range(self._model.njnt):
                if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    # Set quaternion part of the free joint
                    self._data.qpos[i : i + 4] = orientation
                    break

        # Reset velocities and accelerations
        self._data.qvel = np.zeros_like(self._data.qvel)
        self._data.qacc = np.zeros_like(self._data.qacc)

        # Re-initialize state
        mujoco.mj_forward(self._model, self._data)
        mujoco.mj_step(self._model, self._data)

    def close(self) -> None:
        """Clean up simulation resources."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception as e:
                logger.error("Error closing viewer: %s", e)
            self._viewer = None
