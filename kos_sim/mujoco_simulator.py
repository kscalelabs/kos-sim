"""Wrapper around MuJoCo simulation."""

import argparse
import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import colorlogging
import mujoco
import mujoco_viewer
import numpy as np
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput
from omegaconf import OmegaConf

from kos_sim import logger


@dataclass
class SimulationConfig:
    dt: float = field(default=0.001)
    sim_decimation: int = field(default=1)
    joint_id_to_name: dict[int, str] = field(default_factory=dict)
    gravity: bool = field(default=True)
    render: bool = field(default=True)
    suspended: bool = field(default=False)


class MujocoSimulator:
    def __init__(
        self,
        model_path: str | Path,
        model_metadata: RobotURDFMetadataOutput,
        config_path: str | Path | None = None,
    ) -> None:
        # Load config or use default
        self._metadata = model_metadata

        config = cast(SimulationConfig, OmegaConf.structured(SimulationConfig))
        if config_path is not None:
            config = OmegaConf.merge(config, OmegaConf.load(config_path))
        self._config = config

        # Load MuJoCo model and initialize data
        logger.info("Loading model from %s", model_path)
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._model.opt.timestep = self._config.dt  # Use dt from config
        self._data = mujoco.MjData(self._model)

        self._gravity = self._config.gravity
        self._suspended = self._config.suspended
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
        self._render_enabled = self._config.render
        if self._render_enabled:
            self._viewer = mujoco_viewer.MujocoViewer(self._model, self._data)
        else:
            self._viewer = mujoco_viewer.MujocoViewer(self._model, self._data, "offscreen")

        # Cache lookups after initialization
        self._sensor_name_to_id = {self._model.sensor(i).name: i for i in range(self._model.nsensor)}
        print("SENSOR_IDS", self._sensor_name_to_id)
        self._actuator_name_to_id = {self._model.actuator(i).name: i for i in range(self._model.nu)}
        print("ACTUATOR_IDS", self._actuator_name_to_id)

        # There is an important distinction between actuator IDs and joint IDs.
        # joint IDs should be at the kos layer, where the canonical IDs are assigned (see docs.kscale.dev)
        # but actuator IDs are at the mujoco layer, where the actuators actually get mapped.
        print("JOINT_ID_TO_NAME", self._config.joint_id_to_name)
        self._joint_id_to_actuator_id = {
            joint_id: self._actuator_name_to_id[name] for joint_id, name in self._config.joint_id_to_name.items()
        }

        # Add control parameters
        self._lock = threading.Lock()
        self._current_commands: dict[str, float] = {}
        self._actuator_states: dict[int, dict[str, float]] = {}

        self._count_lowlevel = 0
        self._target_positions: dict[str, float] = {}  # Store target positions between updates

        self.pd_counter = 0

    def step(self) -> None:
        """Execute one step of the simulation."""
        with self._lock:
            # Only update commands every sim_decimation steps
            if self._count_lowlevel % self._config.sim_decimation == 0:
                self._target_positions = self._current_commands.copy()

            # Apply position control commands directly to actuators
            for name, target_pos in self._target_positions.items():
                actuator_id = self._actuator_name_to_id[name]
                self._data.ctrl[actuator_id] = target_pos

        # Step physics
        mujoco.mj_step(self._model, self._data)

        # Increment counter
        self._count_lowlevel += 1

        if self._suspended:
            # Find the root joint (floating_base)
            for i in range(self._model.njnt):
                if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    self._data.qpos[i : i + 7] = self._model.keyframe("default").qpos[i : i + 7]
                    self._data.qvel[i : i + 6] = 0
                    break

        return self._data

    def render(self) -> None:
        if self._render_enabled:
            self._viewer.render()

    def get_sensor_data(self, name: str) -> np.ndarray:
        """Get data from a named sensor."""
        with self._lock:
            if name not in self._sensor_name_to_id:
                raise KeyError(f"Sensor '{name}' not found")
            sensor_id = self._sensor_name_to_id[name]
        return self._data.sensor(sensor_id).data.copy()

    def get_actuator_state(self, joint_id: int) -> float:
        """Get current state of an actuator using real joint ID."""
        with self._lock:
            if joint_id not in self._config.joint_id_to_name:
                raise KeyError(f"Joint ID {joint_id} not found in config mappings")

            joint_name = self._config.joint_id_to_name[joint_id]
            if joint_name not in self._actuator_name_to_id:
                raise KeyError(f"Joint {joint_name} not found in MuJoCo model")

            actuator_id = self._actuator_name_to_id[joint_name]
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
                if joint_name not in self._actuator_name_to_id:
                    logger.warning("Joint %s not found in MuJoCo model", joint_name)
                    continue

                self._current_commands[joint_name] = command

    def configure_actuator(self, joint_id: int, configuration: dict[str, float]) -> None:
        actuator_id = self._joint_id_to_actuator_id[joint_id]
        if actuator_id not in self._actuator_states:
            self._actuator_states[actuator_id] = {}
        self._actuator_states[actuator_id].update(configuration)

        if "kp" in configuration:
            self._model.actuator_gainprm[actuator_id, 0] = configuration["kp"]
            logger.info("Set kp for actuator %s to %f", joint_id, configuration["kp"])
        if "kd" in configuration:
            self._model.actuator_biasprm[actuator_id, 2] = configuration["kd"]
            logger.info("Set kd for actuator %s to %f", joint_id, configuration["kd"])

        torque_enabled = configuration.get("torque_enabled", False)
        if not torque_enabled:
            self._actuator_states[actuator_id]["saved_min_tau"] = self._model.actuator_forcerange[actuator_id, 0]
            self._actuator_states[actuator_id]["saved_max_tau"] = self._model.actuator_forcerange[actuator_id, 1]
            self._model.actuator_forcerange[actuator_id, 0] = 0
            self._model.actuator_forcerange[actuator_id, 1] = 0
            logger.info("Set torque_enabled for actuator %s to %s", joint_id, torque_enabled)
        else:
            # if previously disabled torque, restore previous values
            if "saved_min_tau" in self._actuator_states[actuator_id]:
                self._model.actuator_forcerange[actuator_id, 0] = self._actuator_states[actuator_id]["saved_min_tau"]
                self._model.actuator_forcerange[actuator_id, 1] = self._actuator_states[actuator_id]["saved_max_tau"]
                logger.info("Set torque_enabled for actuator %s to %s", joint_id, torque_enabled)

        if "max_torque" in configuration and torque_enabled:
            self._model.actuator_forcerange[actuator_id, 0] = -configuration["max_torque"]
            self._model.actuator_forcerange[actuator_id, 1] = configuration["max_torque"]
            logger.info("Set max_torque for actuator %s to +/-%f", joint_id, configuration["max_torque"])

    def reset(
        self,
        qpos: list[float] | None = None,
    ) -> None:
        """Reset simulation to specified or default state.

        Args:
            base_position: [x, y, z] position for the base
            base_orientation: Quaternion [w, x, y, z] for base orientation
            joint_positions: Dict of joint names to positions (radians)
        """
        logger.info("Resetting simulation")
        logger.info("qpos: %s", qpos)
        mujoco.mj_resetData(self._model, self._data)

        # reset qpos
        if qpos is not None:
            self._data.qpos = qpos

        # Reset velocities and accelerations
        self._data.qvel = np.zeros_like(self._data.qvel)
        self._data.qacc = np.zeros_like(self._data.qacc)

        # Re-initialize state
        mujoco.mj_forward(self._model, self._data)

    def close(self) -> None:
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


async def test_simulation_adhoc(
    model_name: str,
    config_path: str | Path,
    duration: float = 5.0,
    speed: float = 1.0,
    render: bool = True,
) -> None:
    api = K()
    model_dir = await api.download_and_extract_urdf(model_name)
    model_path = next(model_dir.glob("*.mjcf"))

    model_metadata = None  # TODO
    simulator = MujocoSimulator(model_path, model_metadata, config_path)

    timestep = simulator.timestep
    initial_update = last_update = asyncio.get_event_loop().time()

    while True:
        current_time = asyncio.get_event_loop().time()
        if current_time - initial_update > duration:
            break

        sim_time = current_time - last_update
        last_update = current_time
        while sim_time > 0:
            simulator.step()
            sim_time -= timestep

        simulator.render()

    simulator.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test the MuJoCo simulation.")
    parser.add_argument("model_name", type=str, help="Name of the model to simulate")
    parser.add_argument("--config_path", type=str, default="cfg/mujoco.yml", help="Path to the simulation config")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration to run simulation (seconds)")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")

    colorlogging.configure()

    args = parser.parse_args()
    await test_simulation_adhoc(
        args.model_name,
        args.config_path,
        duration=args.duration,
        speed=args.speed,
        render=not args.no_render,
    )


if __name__ == "__main__":
    # python -m kos_sim.simulator
    asyncio.run(main())
    asyncio.run(main())
