"""Server and simulation loop for KOS."""

import argparse
import threading
import time
from concurrent import futures

import colorlogging
import grpc
from kos_protos import actuator_pb2_grpc, imu_pb2_grpc, sim_pb2_grpc

from kos_sim import logger
from kos_sim.config import SimulatorConfig
from kos_sim.services import ActuatorService, IMUService, SimService
from kos_sim.simulator import MujocoSimulator
from kos_sim.stepping import StepController, StepMode


class SimulationServer:
    def __init__(
        self,
        model_path: str,
        config_path: str | None = None,
        port: int = 50051,
        step_mode: StepMode = StepMode.CONTINUOUS,
    ) -> None:
        if config_path:
            config = SimulatorConfig.from_file(config_path)
        else:
            config = SimulatorConfig.default()

        self.simulator = MujocoSimulator(model_path, config=config, render=True)
        self.step_controller = StepController(self.simulator, mode=step_mode)
        self.port = port
        self._stop_event = threading.Event()
        self._grpc_thread: threading.Thread | None = None
        self._server = None

    def _grpc_server_loop(self) -> None:
        """Run the gRPC server in a separate thread."""
        # Create gRPC server
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        assert self._server is not None

        # Add our services
        actuator_service = ActuatorService(self.simulator)
        imu_service = IMUService(self.simulator)
        sim_service = SimService(self.simulator)

        actuator_pb2_grpc.add_ActuatorServiceServicer_to_server(actuator_service, self._server)
        imu_pb2_grpc.add_IMUServiceServicer_to_server(imu_service, self._server)
        sim_pb2_grpc.add_SimulationServiceServicer_to_server(sim_service, self._server)

        # Start the server
        self._server.add_insecure_port(f"[::]:{self.port}")
        self._server.start()
        logger.info("Server started on port %d", self.port)

        # Wait for termination
        self._server.wait_for_termination()

    def start(self) -> None:
        """Start the gRPC server and run simulation in main thread."""
        self._grpc_thread = threading.Thread(target=self._grpc_server_loop)
        self._grpc_thread.start()

        try:
            while not self._stop_event.is_set():
                process_start = time.time()

                if self.step_controller.should_step():
                    self.simulator.step()

                sleep_time = max(0, self.simulator._config.dt - (time.time() - process_start))
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Stop the simulation and cleanup resources."""
        logger.info("Shutting down simulation...")
        self._stop_event.set()
        if self._server is not None:
            self._server.stop(0)
        if self._grpc_thread is not None:
            self._grpc_thread.join()
        self.simulator.close()


def serve(model_path: str, config_path: str | None = None, port: int = 50051) -> None:
    server = SimulationServer(model_path, config_path=config_path, port=port)
    server.start()


def run_server() -> None:
    parser = argparse.ArgumentParser(description="Start the simulation gRPC server.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to MuJoCo XML model file")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--config-path", type=str, default=None, help="Path to config file")

    colorlogging.configure()

    args = parser.parse_args()
    serve(args.model_path, args.config_path, args.port)


if __name__ == "__main__":
    run_server()
