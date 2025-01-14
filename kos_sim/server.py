"""Server and simulation loop for KOS."""

import argparse
import logging
import threading
import time
from concurrent import futures

import grpc
from kos_protos import actuator_pb2_grpc, imu_pb2_grpc

from kos_sim.services import ActuatorService, IMUService
from kos_sim.simulator import MujocoSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulationServer:
    def __init__(self, model_path: str, port: int = 50051, dt: float = 0.002) -> None:
        self.simulator = MujocoSimulator(model_path, dt=dt, render=True, suspended=False)
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

        actuator_pb2_grpc.add_ActuatorServiceServicer_to_server(actuator_service, self._server)
        imu_pb2_grpc.add_IMUServiceServicer_to_server(imu_service, self._server)

        # Start the server
        self._server.add_insecure_port(f"[::]:{self.port}")
        self._server.start()
        logger.info("Server started on port %d", self.port)

        # Wait for termination
        self._server.wait_for_termination()

    def start(self) -> None:
        """Start the gRPC server and run simulation in main thread."""
        # Start gRPC server in separate thread
        self._grpc_thread = threading.Thread(target=self._grpc_server_loop)
        self._grpc_thread.start()

        # Run simulation in main thread
        try:
            while not self._stop_event.is_set():
                self.simulator.step()
                time.sleep(self.simulator._model.opt.timestep)
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


def serve(model_path: str, port: int = 50051) -> None:
    server = SimulationServer(model_path, port)
    server.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the simulation gRPC server.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to MuJoCo XML model file")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")

    args = parser.parse_args()
    serve(args.model_path, args.port)
