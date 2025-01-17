"""Test script for the simulation server."""

import argparse
import math
import time

import pykos

from kos_sim import logger


def test_sim_commands(host: str = "localhost", port: int = 50051) -> None:
    """Test simulation commands."""
    kos = pykos.KOS(ip=host, port=port)
    sim = kos.simulation

    # Test parameters
    frequency = 1.0  # Hz
    _amplitude = 45.0  # degrees
    _duration = 5.0  # seconds

    # Send command
    sim.set_parameters(time_scale=frequency)


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
            logger.info("Command: %.2f, Current: %.2f", position, state.states[0].position)

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error("Test failed: %s", e)


def test_sim_service(host: str = "localhost", port: int = 50051) -> None:
    """Test simulation service commands."""
    kos = pykos.KOS(ip=host, port=port)
    sim = kos.sim

    logger.info("Starting simulation service test...")

    try:
        # Test reset with initial state
        initial_state = {"qpos": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}
        sim.reset(initial_state=initial_state)
        logger.info("Reset simulation with initial state")

        # Test get/set parameters
        sim.set_parameters(time_scale=2.0, gravity=9.81)
        params = sim.get_parameters()
        logger.info(
            "Set parameters - time_scale: %.2f, gravity: %.2f", params.parameters.time_scale, params.parameters.gravity
        )

        # Test pause/unpause and stepping
        sim.set_paused(True)
        logger.info("Paused simulation")

        sim.step(num_steps=100, step_size=0.01)
        logger.info("Stepped simulation 100 steps")

        sim.set_paused(False)
        logger.info("Resumed simulation")

    except Exception as e:
        logger.error("Test failed: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the simulation server with actuator commands.")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--test", type=str, choices=["actuator", "sim"], default="actuator", help="Test to run")

    args = parser.parse_args()
    if args.test == "actuator":
        test_actuator_commands(args.host, args.port)
    else:
        test_sim_service(args.host, args.port)


if __name__ == "__main__":
    main()
