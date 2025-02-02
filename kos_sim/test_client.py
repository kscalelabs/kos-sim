"""Test script for the simulation server (async version)."""

import argparse
import asyncio
import math
import time
import traceback
import pykos

from kos_sim import logger


async def test_sim_commands(host: str = "localhost", port: int = 50051) -> None:
    """Test simulation commands asynchronously."""
    async with pykos.KOS(ip=host, port=port) as kos:
        sim = kos.simulation

        # Test parameters
        frequency = 1.0  # Hz

        # Asynchronously send command
        await sim.set_parameters(time_scale=frequency)
        logger.info("Simulation parameters set: time_scale=%.2f", frequency)


async def test_actuator_commands(host: str = "localhost", port: int = 50051) -> None:
    """Test actuator commands by sending sinusoidal position commands asynchronously."""
    async with pykos.KOS(ip=host, port=port) as kos:
        actuator = kos.actuator

        # Test parameters
        frequency = 1.0  # Hz
        amplitude = 45.0  # degrees
        duration = 5.0  # seconds
        actuator_id = 2

        logger.info("Starting actuator command test...")

        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                # Calculate desired position
                t = time.time() - start_time
                position = amplitude * math.sin(2 * math.pi * frequency * t)

                # Send command
                actuator_commands = [{"actuator_id": actuator_id, "position": position}]
                await actuator.command_actuators(actuator_commands)

                # Get and print current state
                state = await actuator.get_actuators_state([actuator_id])
                logger.info("Command: %.2f, Current: %.2f", position, state.states[0].position)

                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            logger.error("Test failed: %s", e)
            print(traceback.format_exc())


async def test_sim_service(host: str = "localhost", port: int = 50051) -> None:
    """Test simulation service commands asynchronously."""
    async with pykos.KOS(ip=host, port=port) as kos:
        sim = kos.sim

        logger.info("Starting simulation service test...")

        try:
            # Test reset with initial state
            # (this qpos is for kbot-v1)
            initial_state = {"qpos": [0.0, 0.0, 1.16620985, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
            await sim.reset(initial_state=initial_state)
            logger.info("Reset simulation with initial state")

            # Test get/set parameters # TODO: set parameters crash simulation
            # await sim.set_parameters(time_scale=1.0, gravity=9.81)
            # params = await sim.get_parameters()
            # logger.info(
            #     "Set parameters - time_scale: %.2f, gravity: %.2f",
            #     params.parameters.time_scale,
            #     params.parameters.gravity,
            # )

            # Test pause/unpause and stepping
            await sim.set_paused(True)
            logger.info("Paused simulation")

            await sim.step(num_steps=100, step_size=0.01)
            logger.info("Stepped simulation 100 steps")

            await sim.set_paused(False)
            logger.info("Resumed simulation")

        except Exception as e:
            logger.error("Test failed: %s", e)
            print(traceback.format_exc())


async def test_configure_and_command_actuators(host: str = "localhost", port: int = 50051) -> None:
    """
    Configure all actuators and send a zero position command asynchronously.

    The configuration parameters (kp, kd, and torque_enabled) are the same for all actuators.
    """
    async with pykos.KOS(ip=host, port=port) as kos:
        # Configure all actuators
        for actuator_id in range(60):
            try:
                await kos.actuator.configure_actuator(
                    actuator_id=actuator_id, kp=100, kd=10, torque_enabled=True
                )
            except Exception as e:
                logger.error("Failed to configure actuator %s: %s", actuator_id, e)

        # Give some time for the configuration to settle
        await asyncio.sleep(1)

        # Build commands: setting position to 0.0 for all actuators
        commands = [{"actuator_id": actuator_id, "position": 0.0} for actuator_id in range(60)]
        await kos.actuator.command_actuators(commands)
        logger.info("Commanded all actuators to 0.0 position")


async def main():
    """Parse command line arguments and run the selected test asynchronously."""
    parser = argparse.ArgumentParser(
        description="Test the simulation server with actuator or simulation commands (async mode)."
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument(
        "--test",
        type=str,
        choices=["actuator", "sim", "configure"],
        default="actuator",
        help="Test to run: actuator, sim, or configure",
    )

    args = parser.parse_args()

    if args.test == "actuator":
        await test_actuator_commands(args.host, args.port)
    elif args.test == "sim":
        await test_sim_service(args.host, args.port)
    elif args.test == "configure":
        await test_configure_and_command_actuators(args.host, args.port)
    else:
        logger.error("Invalid test option provided.")


if __name__ == "__main__":
    asyncio.run(main())