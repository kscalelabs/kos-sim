"""Server and simulation loop for KOS."""

import argparse
import asyncio
import itertools
import logging
import time
import traceback
from concurrent import futures
from pathlib import Path

import colorlogging
from dataclasses import dataclass, field
import grpc
from kos_protos import actuator_pb2_grpc, imu_pb2_grpc, sim_pb2_grpc
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput
from mujoco_scenes.mjcf import list_scenes

from kos_sim import logger
from kos_sim.services import ActuatorService, IMUService, SimService
from kos_sim.simulator import MujocoSimulator
from kos_sim.utils import get_sim_artifacts_path


class SimulationServer:
    def __init__(
        self,
        model_path: str | Path,
        model_metadata: RobotURDFMetadataOutput,
        host: str = "localhost",
        port: int = 50051,
        dt: float = 0.0001,
        gravity: bool = True,
        render: bool = True,
        suspended: bool = False,
        command_delay_min: float = 0.0,
        command_delay_max: float = 0.0,
        sleep_time: float = 1e-6,
        mujoco_scene: str = "smooth",
        render_decimation: int = 15,
    ) -> None:
        self.simulator = MujocoSimulator(
            model_path=model_path,
            model_metadata=model_metadata,
            dt=dt,
            gravity=gravity,
            render=render,
            suspended=suspended,
            command_delay_min=command_delay_min,
            command_delay_max=command_delay_max,
            mujoco_scene=mujoco_scene,
            render_decimation=render_decimation,
        )
        self.host = host
        self.port = port
        self._sleep_time = sleep_time
        self._stop_event = asyncio.Event()
        self._server = None

    async def _grpc_server_loop(self) -> None:
        """Run the async gRPC server."""
        # Create async gRPC server
        self._server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))

        assert self._server is not None

        # Add our services (these need to be modified to be async as well)
        actuator_service = ActuatorService(self.simulator)
        imu_service = IMUService(self.simulator)
        sim_service = SimService(self.simulator)

        actuator_pb2_grpc.add_ActuatorServiceServicer_to_server(actuator_service, self._server)
        imu_pb2_grpc.add_IMUServiceServicer_to_server(imu_service, self._server)
        sim_pb2_grpc.add_SimulationServiceServicer_to_server(sim_service, self._server)

        # Start the server
        self._server.add_insecure_port(f"{self.host}:{self.port}")
        await self._server.start()
        logger.info("Server started on %s:%d", self.host, self.port)
        await self._server.wait_for_termination()

    async def simulation_loop(self) -> None:
        """Run the simulation loop asynchronously."""
        start_time = time.time()
        num_renders = 0
        total_steps = 0
        try:
            while not self._stop_event.is_set():
                while self.simulator._sim_time < time.time():
                    # Run one control loop.
                    for _ in range(self.simulator._sim_decimation):
                        await self.simulator.step()
                    await asyncio.sleep(self._sleep_time)

                if total_steps % self.simulator._render_decimation == 0:
                    await self.simulator.render()
                    num_renders += 1


                # Sleep until the next control update.
                current_time = time.time()
                if current_time < self.simulator._sim_time:
                    await asyncio.sleep(self.simulator._sim_time - current_time)

                total_steps += 1
                logger.debug(
                    "Simulation time: %f, rendering frequency: %f",
                    self.simulator._sim_time,
                    num_renders / (time.time() - start_time),
                )

        except Exception as e:
            logger.error("Simulation loop failed: %s", e)
            logger.error("Traceback: %s", traceback.format_exc())

        finally:
            await self.stop()

    async def start(self) -> None:
        """Start both the gRPC server and simulation loop asynchronously."""
        grpc_task = asyncio.create_task(self._grpc_server_loop())
        sim_task = asyncio.create_task(self.simulation_loop())

        try:
            await asyncio.gather(grpc_task, sim_task)
        except asyncio.CancelledError:
            await self.stop()

    async def stop(self) -> None:
        """Stop the simulation and cleanup resources asynchronously."""
        logger.info("Shutting down simulation...")
        self._stop_event.set()
        if self._server is not None:
            await self._server.stop(0)
        await self.simulator.close()


async def get_model_metadata(api: K, model_name: str) -> RobotURDFMetadataOutput:
    model_path = get_sim_artifacts_path() / model_name / "metadata.json"
    if model_path.exists():
        return RobotURDFMetadataOutput.model_validate_json(model_path.read_text())
    model_path.parent.mkdir(parents=True, exist_ok=True)
    robot_class = await api.get_robot_class(model_name)
    metadata = robot_class.metadata
    if metadata is None:
        raise ValueError(f"No metadata found for model {model_name}")
    model_path.write_text(metadata.model_dump_json())
    return metadata


async def serve(
    model_name: str,
    host: str = "localhost",
    port: int = 50051,
    dt: float = 0.001,
    gravity: bool = True,
    render: bool = True,
    suspended: bool = False,
    command_delay_min: float = 0.0,
    command_delay_max: float = 0.0,
    mujoco_scene: str = "smooth",
    model_path: str | Path | None = None,
) -> None:
    # TODO - remove this
    if model_path is None:
        async with K() as api:
            model_dir, model_metadata = await asyncio.gather(
                api.download_and_extract_urdf(model_name),
                get_model_metadata(api, model_name),
            )

        model_path = next(
            itertools.chain(
                model_dir.glob("*.mjcf"),
                model_dir.glob("*.xml"),
            )
        )
    else:
        @dataclass
        class JointMetadata:
            id: int
            kp: float
            kd: float

        @dataclass
        class ModelInfo:
            joint_name_to_metadata: dict
            control_frequency: float = 50.0

    @dataclass
    class JointMetadata:
        id: int
        kp: float
        kd: float

    @dataclass
    class ModelInfo:
        joint_name_to_metadata: dict
        control_frequency: float = 50.0

    model_metadata = ModelInfo(
        joint_name_to_metadata={
            "left_hip_pitch_04": JointMetadata(id=31, kp=300.0, kd=5.0),
            "left_hip_roll_03": JointMetadata(id=32, kp=120.0, kd=5.0),
            "left_hip_yaw_03": JointMetadata(id=33, kp=120.0, kd=5.0),
            "left_knee_04": JointMetadata(id=34, kp=300.0, kd=5.0),
            "left_ankle_02": JointMetadata(id=35, kp=40.0, kd=5.0),
            "right_hip_pitch_04": JointMetadata(id=41, kp=300.0, kd=5.0),
            "right_hip_roll_03": JointMetadata(id=42, kp=120.0, kd=5.0),
            "right_hip_yaw_03": JointMetadata(id=43, kp=120.0, kd=5.0),
            "right_knee_04": JointMetadata(id=44, kp=300.0, kd=5.0),
            "right_ankle_02": JointMetadata(id=45, kp=40.0, kd=5.0),
        }
    )

    server = SimulationServer(
        model_path,
        model_metadata=model_metadata,
        host=host,
        port=port,
        dt=dt,
        gravity=gravity,
        render=render,
        suspended=suspended,
        command_delay_min=command_delay_min,
        command_delay_max=command_delay_max,
        mujoco_scene=mujoco_scene,
    )
    await server.start()


async def run_server() -> None:
    parser = argparse.ArgumentParser(description="Start the simulation gRPC server.")
    parser.add_argument("model_name", type=str, help="Name of the model to simulate")
    parser.add_argument("--host", type=str, default="localhost", help="Host to listen on")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--dt", type=float, default=0.001, help="Simulation timestep")
    parser.add_argument("--no-gravity", action="store_true", help="Disable gravity")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--suspended", action="store_true", help="Suspended simulation")
    parser.add_argument("--command-delay-min", type=float, default=0.0, help="Minimum command delay")
    parser.add_argument("--command-delay-max", type=float, default=0.0, help="Maximum command delay")
    parser.add_argument("--scene", choices=list_scenes(), default="smooth", help="Mujoco scene to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--model-path",  type=str, help="Path to the model to simulate")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    model_name = args.model_name
    host = args.host
    port = args.port
    dt = args.dt
    gravity = not args.no_gravity
    render = not args.no_render
    suspended = args.suspended
    command_delay_min = args.command_delay_min
    command_delay_max = args.command_delay_max
    mujoco_scene = args.scene

    logger.info("Model name: %s", model_name)
    logger.info("Port: %d", port)
    logger.info("DT: %f", dt)
    logger.info("Gravity: %s", gravity)
    logger.info("Render: %s", render)
    logger.info("Suspended: %s", suspended)
    logger.info("Command delay min: %f", command_delay_min)
    logger.info("Command delay max: %f", command_delay_max)
    logger.info("Mujoco scene: %s", mujoco_scene)

    await serve(
        model_name=model_name,
        host=host,
        port=port,
        dt=dt,
        gravity=gravity,
        render=render,
        suspended=suspended,
        command_delay_min=command_delay_min,
        command_delay_max=command_delay_max,
        mujoco_scene=mujoco_scene,
        model_path=args.model_path,
    )


def main() -> None:
    asyncio.run(run_server())


if __name__ == "__main__":
    # python -m kos_sim.server
    main()
