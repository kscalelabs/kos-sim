"""Wrapper around MuJoCo simulation."""

import argparse
import asyncio
import threading
import yaml
from pathlib import Path

import colorlogging
import mujoco
import mujoco_viewer
import numpy as np
from dataclasses import dataclass
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput

from kos_sim import logger


@dataclass
class SimulationConfig:
    dt: float
    kp: float
    kd: float
    sim_decimation: int


class MujocoSimulator:
    def __init__(
        self,
        model_path: str | Path,
        model_metadata: RobotURDFMetadataOutput,
        config_path: str | Path,
        gravity: bool = True,
        render: bool = True,
        suspended: bool = False,
    ) -> None:
        # Load config or use default
        self._metadata = model_metadata

        # config yml load as SimulationConfig
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        self._config = SimulationConfig(**config_data)

        # Load MuJoCo model and initialize data
        print("Loading model from %s", model_path)
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._model.opt.timestep = self._config.dt  # Use dt from config
        self._data = mujoco.MjData(self._model)

        self._gravity = gravity
        self._suspended = suspended
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
                    self._data.qpos[i : i + 7] = self._model.keyframe("default").qpos[i : i + 7]
                    self._data.qvel[i : i + 6] = 0
                    break

        return self._data

    def render(self) -> None:
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
    model_name: str, config_path: str | Path, duration: float = 5.0, speed: float = 1.0, render: bool = True
) -> None:
    api = K()
    model_dir = await api.download_and_extract_urdf(model_name)
    model_path = next(model_dir.glob("*.mjcf"))

    simulator = MujocoSimulator(model_path, config_path, render=render)

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
