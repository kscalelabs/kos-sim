"""Server and simulation loop for KOS."""

import argparse
import asyncio
import itertools
import json
import logging
import os
import time
import traceback
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import colorlogging
import grpc
from kos_protos import actuator_pb2_grpc, imu_pb2_grpc, process_manager_pb2_grpc, sim_pb2_grpc
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput
from mujoco_scenes.mjcf import list_scenes

from kos_sim import logger
from kos_sim.services import ActuatorService, IMUService, ProcessManagerService, SimService
from kos_sim.simulator import MujocoSimulator
from kos_sim.utils import get_sim_artifacts_path
from kos_sim.video_recorder import VideoRecorder


@dataclass
class SimulationRandomizationConfig:
    command_delay_min: float = 0.0
    command_delay_max: float = 0.0
    joint_pos_delta_noise: float = 0.0
    joint_pos_noise: float = 0.0
    joint_vel_noise: float = 0.0


@dataclass
class RenderingConfig:
    render: bool = True
    render_frequency: float = 1
    video_output_dir: str | Path | None = None
    frame_width: int = 640
    frame_height: int = 480
    camera: str | None = None


@dataclass
class PhysicsConfig:
    gravity: bool = True
    suspended: bool = False
    start_height: float = 1.5
    dt: float = 0.0001


@dataclass
class SimulationServerConfig:
    randomization: SimulationRandomizationConfig
    rendering: RenderingConfig
    physics: PhysicsConfig
    model_path: str | Path
    model_metadata: RobotURDFMetadataOutput
    actuator_params_path: str | Path
    mujoco_scene: str = "smooth"
    host: str = "localhost"
    port: int = 50051
    sleep_time: float = 1e-6
    fixed_base: bool = False


class SimulationServer:
    def __init__(
        self,
        config: SimulationServerConfig,
    ) -> None:
        self.simulator = MujocoSimulator(
            model_path=config.model_path,
            model_metadata=config.model_metadata,
            actuator_params_path=config.actuator_params_path,
            dt=config.physics.dt,
            gravity=config.physics.gravity,
            render_mode="window" if config.rendering.render else "offscreen",
            suspended=config.physics.suspended,
            start_height=config.physics.start_height,
            command_delay_min=config.randomization.command_delay_min,
            command_delay_max=config.randomization.command_delay_max,
            joint_pos_delta_noise=config.randomization.joint_pos_delta_noise,
            joint_pos_noise=config.randomization.joint_pos_noise,
            joint_vel_noise=config.randomization.joint_vel_noise,
            mujoco_scene=config.mujoco_scene,
            camera=config.rendering.camera,
            frame_width=config.rendering.frame_width,
            frame_height=config.rendering.frame_height,
            fixed_base=config.fixed_base,
        )
        self.host = config.host
        self.port = config.port
        self._sleep_time = config.sleep_time
        self._stop_event = asyncio.Event()
        self._server = None
        self._step_lock = asyncio.Semaphore(1)
        self._render_decimation = int(1.0 / config.rendering.render_frequency)

        # Initialize video recorder if needed
        self.video_recorder = None
        if config.rendering.video_output_dir is not None:
            self.video_recorder = VideoRecorder(
                simulator=self.simulator,
                output_dir=config.rendering.video_output_dir,
                fps=int(self.simulator._control_frequency),
                frame_width=config.rendering.frame_width,
                frame_height=config.rendering.frame_height,
            )

    async def _grpc_server_loop(self) -> None:
        """Run the async gRPC server."""
        # Create async gRPC server
        self._server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))

        assert self._server is not None

        # Add our services (these need to be modified to be async as well)
        actuator_service = ActuatorService(self.simulator, self._step_lock)
        imu_service = IMUService(self.simulator)
        sim_service = SimService(self.simulator)
        process_manager_service = ProcessManagerService(self.simulator, self.video_recorder)

        actuator_pb2_grpc.add_ActuatorServiceServicer_to_server(actuator_service, self._server)
        imu_pb2_grpc.add_IMUServiceServicer_to_server(imu_service, self._server)
        sim_pb2_grpc.add_SimulationServiceServicer_to_server(sim_service, self._server)
        process_manager_pb2_grpc.add_ProcessManagerServiceServicer_to_server(process_manager_service, self._server)

        # Start the server
        self._server.add_insecure_port(f"{self.host}:{self.port}")
        await self._server.start()
        logger.info("Server started on %s:%d", self.host, self.port)
        await self._server.wait_for_termination()

    async def simulation_loop(self) -> None:
        """Run the simulation loop asynchronously."""
        start_time = time.time()
        num_renders = 0
        num_steps = 0

        try:
            while not self._stop_event.is_set():
                while self.simulator._sim_time < time.time():
                    # Run one control loop.
                    async with self._step_lock:
                        for _ in range(self.simulator._sim_decimation):
                            await self.simulator.step()
                    await asyncio.sleep(self._sleep_time)

                if num_steps % self._render_decimation == 0:
                    await self.simulator.render()
                    num_renders += 1

                if self.video_recorder is not None and self.video_recorder.is_recording:
                    await self.video_recorder.capture_frame()

                # Sleep until the next control update.
                current_time = time.time()
                if current_time < self.simulator._sim_time:
                    await asyncio.sleep(self.simulator._sim_time - current_time)
                num_steps += 1
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
        if self.video_recorder is not None and self.video_recorder.is_recording:
            self.video_recorder.stop_recording()
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
    rendering: RenderingConfig = RenderingConfig(),
    physics: PhysicsConfig = PhysicsConfig(),
    randomization: SimulationRandomizationConfig = SimulationRandomizationConfig(),
    mujoco_scene: str = "smooth",
    local_assets: bool = False,
    fixed_base: bool = False,
) -> None:
    if local_assets:
        kscale_assets_path = os.getenv("KSCALE_ASSETS_PATH")
        if not kscale_assets_path:
            logger.error("KSCALE_ASSETS_PATH environment variable must be set when using --load-assets")
            raise SystemExit(1)
        
        local_model_dir = Path(kscale_assets_path) / model_name
        urdf_files = list(local_model_dir.glob("*.mjcf")) or list(local_model_dir.glob("*.xml"))
        metadata_file = local_model_dir / "metadata.json"
        actuator_params_path = Path(kscale_assets_path) / "actuators"
        
        if not (urdf_files and metadata_file.exists()):
            logger.error(f"Required files not found in {local_model_dir}")
            logger.error("Expected .mjcf/.xml file and metadata.json")
            raise SystemExit(1)
            
        logger.info(f"Loading assets from local path: {local_model_dir}")
        logger.info(f"Loading actuator params from: {actuator_params_path}")
        with open(metadata_file, "r") as f:
            model_metadata = RobotURDFMetadataOutput.model_validate_json(metadata_file.read_text())
        model_dir = local_model_dir
    else:
        logger.info("Loading assets from KScale API")
        actuator_params_path = ""
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

    config = SimulationServerConfig(
        model_path=model_path,
        model_metadata=model_metadata,
        actuator_params_path=actuator_params_path,
        physics=physics,
        rendering=rendering,
        randomization=randomization,
        fixed_base=fixed_base,
    )

    server = SimulationServer(config)
    await server.start()


async def run_server() -> None:
    parser = argparse.ArgumentParser(description="Start the simulation gRPC server.")
    parser.add_argument("model_name", type=str, help="Name of the model to simulate")
    parser.add_argument("--host", type=str, default="localhost", help="Host to listen on")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--dt", type=float, default=0.002, help="Simulation timestep")
    parser.add_argument("--no-gravity", action="store_true", help="Disable gravity")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--render-frequency", type=float, default=1, help="Render frequency (Hz)")
    parser.add_argument("--suspended", action="store_true", help="Suspended simulation")
    parser.add_argument("--command-delay-min", type=float, default=0.0, help="Minimum command delay")
    parser.add_argument("--command-delay-max", type=float, default=0.0, help="Maximum command delay")
    parser.add_argument("--start-height", type=float, default=1.5, help="Start height")
    parser.add_argument("--joint-pos-delta-noise", type=float, default=0.0, help="Joint position delta noise (degrees)")
    parser.add_argument("--joint-pos-noise", type=float, default=0.0, help="Joint position noise (degrees)")
    parser.add_argument("--joint-vel-noise", type=float, default=0.0, help="Joint velocity noise (degrees/second)")
    parser.add_argument("--scene", choices=list_scenes(), default="smooth", help="Mujoco scene to use")
    parser.add_argument("--camera", type=str, default=None, help="Camera to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--video-output-dir", type=str, default="videos", help="Directory to save videos")
    parser.add_argument("--frame-width", type=int, default=640, help="Frame width")
    parser.add_argument("--frame-height", type=int, default=480, help="Frame height")
    parser.add_argument("--local-assets", action="store_true", help="Load assets from KSCALE_ASSETS_PATH")
    parser.add_argument("--fixed-base", action="store_true", help="Use fixed base")

    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    model_name = args.model_name
    host = args.host
    port = args.port
    dt = args.dt
    gravity = not args.no_gravity
    render = not args.no_render
    render_frequency = args.render_frequency
    suspended = args.suspended
    start_height = args.start_height
    command_delay_min = args.command_delay_min
    command_delay_max = args.command_delay_max
    joint_pos_delta_noise = args.joint_pos_delta_noise
    joint_pos_noise = args.joint_pos_noise
    joint_vel_noise = args.joint_vel_noise
    mujoco_scene = args.scene
    camera = args.camera
    frame_width = args.frame_width
    frame_height = args.frame_height

    video_output_dir = args.video_output_dir if not render else None

    logger.info("Model name: %s", model_name)
    logger.info("Port: %d", port)
    logger.info("DT: %f", dt)
    logger.info("Gravity: %s", gravity)
    logger.info("Render: %s", render)
    logger.info("Render frequency: %f", render_frequency)
    logger.info("Suspended: %s", suspended)
    logger.info("Start height: %f", start_height)
    logger.info("Command delay min: %f", command_delay_min)
    logger.info("Command delay max: %f", command_delay_max)
    logger.info("Joint pos delta noise: %f", joint_pos_delta_noise)
    logger.info("Joint pos noise: %f", joint_pos_noise)
    logger.info("Joint vel noise: %f", joint_vel_noise)
    logger.info("Mujoco scene: %s", mujoco_scene)
    logger.info("Camera: %s", camera)
    logger.info("Video output dir: %s", video_output_dir)
    logger.info("Frame width: %d", frame_width)
    logger.info("Frame height: %d", frame_height)

    if video_output_dir is not None:
        os.makedirs(video_output_dir, exist_ok=True)

    rendering = RenderingConfig(
        render=render,
        render_frequency=render_frequency,
        video_output_dir=video_output_dir,
        frame_width=frame_width,
        frame_height=frame_height,
        camera=camera,
    )

    physics = PhysicsConfig(
        dt=dt,
        gravity=gravity,
        suspended=suspended,
        start_height=start_height,
    )

    randomization = SimulationRandomizationConfig(
        command_delay_min=command_delay_min,
        command_delay_max=command_delay_max,
        joint_pos_delta_noise=joint_pos_delta_noise,
        joint_pos_noise=joint_pos_noise,
        joint_vel_noise=joint_vel_noise,
    )

    await serve(
        model_name=model_name,
        host=host,
        port=port,
        rendering=rendering,
        physics=physics,
        randomization=randomization,
        mujoco_scene=mujoco_scene,
        local_assets=args.local_assets,
        fixed_base=args.fixed_base,
    )


def main() -> None:
    asyncio.run(run_server())


if __name__ == "__main__":
    # python -m kos_sim.server
    main()
