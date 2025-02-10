"""Server and simulation loop for KOS."""

import argparse
import asyncio
import logging
import time
import traceback
from concurrent import futures
from pathlib import Path

import colorlogging
import grpc
from kos_protos import actuator_pb2_grpc, imu_pb2_grpc, sim_pb2_grpc
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput

from kos_sim import logger
from kos_sim.services import ActuatorService, IMUService, SimService
from kos_sim.simulator import MujocoSimulator
from kos_sim.stepping import StepController, StepMode
from kos_sim.utils import get_sim_artifacts_path


class SimulationServer:
    def __init__(
        self,
        model_path: str | Path,
        model_metadata: RobotURDFMetadataOutput,
        host: str = "localhost",
        port: int = 50051,
        step_mode: StepMode = StepMode.CONTINUOUS,
        dt: float = 0.001,
        gravity: bool = True,
        render: bool = True,
        suspended: bool = False,
    ) -> None:
        self.simulator = MujocoSimulator(
            model_path=model_path,
            model_metadata=model_metadata,
            dt=dt,
            gravity=gravity,
            render=render,
            suspended=suspended,
        )
        self.step_controller = StepController(self.simulator, mode=step_mode)
        self.host = host
        self.port = port
        self._stop_event = asyncio.Event()
        self._server = None

    async def _grpc_server_loop(self) -> None:
        """Run the async gRPC server."""
        # Create async gRPC server
        self._server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))

        assert self._server is not None

        # Add our services (these need to be modified to be async as well)
        actuator_service = ActuatorService(self.simulator, self.step_controller)
        imu_service = IMUService(self.simulator, self.step_controller)
        sim_service = SimService(self.simulator, self.step_controller)

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
        last_update = time.time()

        try:
            while not self._stop_event.is_set():
                current_time = time.time()
                sim_time = current_time - last_update
                last_update = current_time

                if await self.step_controller.should_step():
                    while sim_time > 0:
                        await self.simulator.step()
                        sim_time -= self.simulator.timestep
                await self.simulator.render()

                # Add a small sleep to prevent the loop from consuming too much CPU.
                await asyncio.sleep(0.001)

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
) -> None:
    async with K() as api:
        model_dir, model_metadata = await asyncio.gather(
            api.download_and_extract_urdf(model_name),
            get_model_metadata(api, model_name),
        )

    model_path = next(model_dir.glob("*.mjcf"))

    server = SimulationServer(
        model_path,
        model_metadata=model_metadata,
        host=host,
        port=port,
        dt=dt,
        gravity=gravity,
        render=render,
        suspended=suspended,
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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    model_name = args.model_name
    host = args.host
    port = args.port
    dt = args.dt
    gravity = not args.no_gravity
    render = not args.no_render
    suspended = args.suspended

    logger.info("Model name: %s", model_name)
    logger.info("Port: %d", port)
    logger.info("DT: %f", dt)
    logger.info("Gravity: %s", gravity)
    logger.info("Render: %s", render)
    logger.info("Suspended: %s", suspended)

    await serve(
        model_name=model_name,
        host=host,
        port=port,
        dt=dt,
        gravity=gravity,
        render=render,
        suspended=suspended,
    )


def main() -> None:
    asyncio.run(run_server())


if __name__ == "__main__":
    # python -m kos_sim.server
    main()
