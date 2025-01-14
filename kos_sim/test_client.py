"""Test script for the simulation server."""

import argparse
import logging
import math
import time

import pykos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_actuator_commands(host: str = "localhost", port: int = 50051) -> None:
    """Test actuator commands by sending sinusoidal position commands."""
    # Connect to simulation server
    kos = pykos.KOS(ip=host, port=port)
    actuator = kos.actuator

    # Test parameters
    frequency = 1.0  # Hz
    amplitude = 45.0  # degrees
    duration = 5.0  # seconds
    actuator_id = 2

    logger.info("Starting actuator command test...")

    try:
        start_time = time.time()
        while time.time() - start_time < duration:
            # Calculate desired position
            t = time.time() - start_time
            position = amplitude * math.sin(2 * math.pi * frequency * t)

            # Send command
            actuator_commands = [{"actuator_id": actuator_id, "position": position}]
            actuator.command_actuators(actuator_commands)

            # Get and print current state
            state = actuator.get_actuators_state([actuator_id])
            logger.info("Command: %.2f, Current: %.2f", position, state[0].position)

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error("Test failed: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the simulation server with actuator commands.")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")

    args = parser.parse_args()
    test_actuator_commands(args.host, args.port)
