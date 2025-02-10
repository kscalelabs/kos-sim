"""Interactive example of the K-Bot simulation."""

import argparse
import asyncio
import logging
import math
import time
from dataclasses import dataclass

import colorlogging
from pykos import KOS

logger = logging.getLogger(__name__)


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float


ACTUATOR_LIST: list[Actuator] = [
    Actuator(11, 1, 150.0, 8.0, 60.0),  # left_shoulder_pitch_03
    Actuator(12, 5, 150.0, 8.0, 60.0),  # left_shoulder_roll_03
    Actuator(13, 9, 50.0, 5.0, 17.0),  # left_shoulder_yaw_02
    Actuator(14, 13, 50.0, 5.0, 17.0),  # left_elbow_02
    Actuator(15, 17, 20.0, 2.0, 17.0),  # left_wrist_02
    Actuator(21, 3, 150.0, 8.0, 60.0),  # right_shoulder_pitch_03
    Actuator(22, 7, 150.0, 8.0, 60.0),  # right_shoulder_roll_03
    Actuator(23, 11, 50.0, 5.0, 17.0),  # right_shoulder_yaw_02
    Actuator(24, 15, 50.0, 5.0, 17.0),  # right_elbow_02
    Actuator(25, 19, 20.0, 2.0, 17.0),  # right_wrist_02
    Actuator(31, 0, 250.0, 30.0, 120.0),  # left_hip_pitch_04
    Actuator(32, 4, 150.0, 8.0, 60.0),  # left_hip_roll_03
    Actuator(33, 8, 150.0, 8.0, 60.0),  # left_hip_yaw_03
    Actuator(34, 12, 200.0, 8.0, 120.0),  # left_knee_04
    Actuator(35, 16, 80.0, 10.0, 17.0),  # left_ankle_02
    Actuator(41, 2, 250.0, 30.0, 120.0),  # right_hip_pitch_04
    Actuator(42, 6, 150.0, 8.0, 60.0),  # right_hip_roll_03
    Actuator(43, 10, 150.0, 8.0, 60.0),  # right_hip_yaw_03
    Actuator(44, 14, 200.0, 8.0, 120.0),  # right_knee_04
    Actuator(45, 18, 80.0, 10.0, 17.0),  # right_ankle_02
]


async def test_client(host: str = "localhost", port: int = 50051) -> None:
    colorlogging.configure(level=logging.DEBUG)
    logger.info("Starting test client...")

    async with KOS(ip=host, port=port) as kos:
        # Reset the simulation.
        await kos.sim.reset()

        await asyncio.sleep(1.0)

        # Configure all actuators
        for actuator in ACTUATOR_LIST:
            await kos.actuator.configure_actuator(
                actuator_id=actuator.actuator_id,
                # kp=actuator.kp,
                # kd=actuator.kd,
                max_torque=actuator.max_torque,
                torque_enabled=True,
            )

        logger.info("Starting control loop...")
        start_time = time.time()

        while True:
            current_time = time.time() - start_time

            # Create commands for all actuators
            commands = []
            for actuator in ACTUATOR_LIST:
                # position = math.sin(2 * math.pi * current_time / 25.0) * math.pi / 4
                position = 0.0
                commands.append(
                    {
                        "actuator_id": actuator.actuator_id,
                        "position": position,
                    }
                )

            # Send commands to all actuators
            logger.info("Sending commands %s", commands)
            await kos.actuator.command_actuators(commands)

            # Run at 50Hz
            await asyncio.sleep(1 / 50)


async def main() -> None:
    """Runs the main simulation loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()

    logger.info("Starting test client...")

    await test_client(
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    # python -m examples.kbot
    asyncio.run(main())
