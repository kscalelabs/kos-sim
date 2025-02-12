"""Interactive example script for a command to keep the robot balanced."""

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass

import colorlogging
import numpy as np
from pykos import KOS
from scipy.spatial.transform import Rotation as R

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
    logger.info("Starting test client...")

    async with KOS(ip=host, port=port) as kos:
        # Configure all actuators
        for actuator in ACTUATOR_LIST:
            await kos.actuator.configure_actuator(
                actuator_id=actuator.actuator_id,
                kp=actuator.kp,
                kd=actuator.kd,
                max_torque=actuator.max_torque,
                torque_enabled=True,
            )

        # Reset the simulation.
        await kos.sim.reset()

        start_time = time.time()
        next_time = start_time + 1 / 50
        delta = 0.0

        while True:
            current_time = time.time()

            _, raw_quat = await asyncio.gather(
                kos.actuator.command_actuators(
                    [
                        # Right leg.
                        {"actuator_id": 41, "position": -30.0 + delta / 2},  # right_hip_pitch_04
                        {"actuator_id": 44, "position": -45.0 + delta},  # right_knee_04
                        {"actuator_id": 45, "position": 20.0 + delta / 2},  # right_ankle_02
                        # Left leg.
                        {"actuator_id": 31, "position": 30.0 - delta / 2},  # left_hip_pitch_04
                        {"actuator_id": 34, "position": 45.0 - delta},  # left_knee_04
                        {"actuator_id": 35, "position": -20.0 - delta / 2},  # left_ankle_02
                    ]
                ),
                kos.imu.get_quaternion(),
            )

            # Gets the direction of gravity. The Z-axis is up.
            quat = R.from_quat([raw_quat.x, raw_quat.y, raw_quat.z, raw_quat.w])
            gravity_direction = quat.apply(np.array([0, 0, -1]))

            # Make the hips move in the opposite direction of gravity.
            scale = gravity_direction[0] - 0.1
            delta = scale * -100.0

            logger.info("Delta: %f", delta)
            if next_time > current_time:
                logger.info("Sleeping for %f seconds", next_time - current_time)
                await asyncio.sleep(next_time - current_time)
            next_time += 1 / 50


async def main() -> None:
    """Runs the main simulation loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)
    await test_client(host=args.host, port=args.port)


if __name__ == "__main__":
    # python -m examples.kbot.balancing
    asyncio.run(main())
