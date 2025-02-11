"""Interactive example script for a command to keep the robot balanced."""

import argparse
import asyncio
import logging
import time

import colorlogging
import numpy as np
from pykos import KOS
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


ACTUATOR_MAPPING = {
    "left_shoulder_yaw": 11,
    "left_shoulder_pitch": 12,
    "left_elbow": 13,
    "left_gripper": 14,
    "right_shoulder_yaw": 21,
    "right_shoulder_pitch": 22,
    "right_elbow": 23,
    "right_gripper": 24,
    "left_hip_yaw": 31,
    "left_hip_roll": 32,
    "left_hip_pitch": 33,
    "left_knee": 34,
    "left_ankle": 35,
    "right_hip_yaw": 41,
    "right_hip_roll": 42,
    "right_hip_pitch": 43,
    "right_knee": 44,
    "right_ankle": 45,
}


async def test_client(host: str = "localhost", port: int = 50051) -> None:
    logger.info("Starting test client...")

    async with KOS(ip=host, port=port) as kos:
        # Reset the simulation.
        await kos.sim.reset()

        # Configure all actuators
        for actuator_id in ACTUATOR_MAPPING.values():
            await kos.actuator.configure_actuator(
                actuator_id=actuator_id,
                torque_enabled=True,
            )

        start_time = time.time()
        next_time = start_time + 1 / 50
        delta = 0.0

        while True:
            current_time = time.time()

            _, raw_quat = await asyncio.gather(
                kos.actuator.command_actuators(
                    [
                        # Right leg.
                        {"actuator_id": 43, "position": 40.0 + delta / 2},  # right_hip_pitch
                        {"actuator_id": 44, "position": -65.0 + delta},  # right_knee
                        {"actuator_id": 45, "position": -30.0 - delta / 2},  # right_ankle
                        # Left leg.
                        {"actuator_id": 33, "position": -40.0 - delta / 2},  # left_hip_pitch
                        {"actuator_id": 34, "position": 65.0 - delta},  # left_knee
                        {"actuator_id": 35, "position": 30.0 + delta / 2},  # left_ankle
                    ]
                ),
                kos.imu.get_quaternion(),
            )

            # Gets the direction of gravity. The Z-axis is up.
            quat = R.from_quat([raw_quat.x, raw_quat.y, raw_quat.z, raw_quat.w])
            gravity_direction = quat.apply(np.array([0, 0, -1]))

            # Make the hips move in the opposite direction of gravity.
            scale = gravity_direction[0]
            delta = scale * -1.0

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
    # python -m examples.zbot.balancing
    asyncio.run(main())
