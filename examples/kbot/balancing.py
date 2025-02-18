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


ACTUATOR_LIST: list[int] = [
    11,  # left_shoulder_pitch_03
    12,  # left_shoulder_roll_03
    13,  # left_shoulder_yaw_02
    14,  # left_elbow_02
    15,  # left_wrist_02
    21,  # right_shoulder_pitch_03
    22,  # right_shoulder_roll_03
    23,  # right_shoulder_yaw_02
    24,  # right_elbow_02
    25,  # right_wrist_02
    31,  # left_hip_pitch_04
    32,  # left_hip_roll_03
    33,  # left_hip_yaw_03
    34,  # left_knee_04
    35,  # left_ankle_02
    41,  # right_hip_pitch_04
    42,  # right_hip_roll_03
    43,  # right_hip_yaw_03
    44,  # right_knee_04
    45,  # right_ankle_02
]


async def test_client(host: str = "localhost", port: int = 50051) -> None:
    logger.info("Starting test client...")

    async with KOS(ip=host, port=port) as kos:
        # Reset the simulation.
        await kos.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})

        start_time = time.time()
        next_time = start_time + 1 / 50
        delta = 0.0

        while True:
            current_time = time.time()

            _, raw_quat = await asyncio.gather(
                kos.actuator.command_actuators(
                    [
                        # Left arm.
                        {"actuator_id": 11, "position": 0.0, "velocity": 0.0},  # left_shoulder_pitch_03
                        {"actuator_id": 12, "position": -20.0, "velocity": 0.0},  # left_shoulder_roll_03
                        {"actuator_id": 13, "position": 0.0, "velocity": 0.0},  # left_shoulder_yaw_02
                        {"actuator_id": 14, "position": 0.0, "velocity": 0.0},  # left_elbow_02
                        {"actuator_id": 15, "position": 0.0, "velocity": 0.0},  # left_wrist_02
                        # Right arm.
                        {"actuator_id": 21, "position": 0.0, "velocity": 0.0},  # right_shoulder_pitch_03
                        {"actuator_id": 22, "position": 20.0, "velocity": 0.0},  # right_shoulder_roll_03
                        {"actuator_id": 23, "position": 0.0, "velocity": 0.0},  # right_shoulder_yaw_02
                        {"actuator_id": 24, "position": 0.0, "velocity": 0.0},  # right_elbow_02
                        {"actuator_id": 25, "position": 0.0, "velocity": 0.0},  # right_wrist_02
                        # Right leg.
                        {"actuator_id": 41, "position": 0.0, "velocity": 0.0},  # right_hip_pitch_04
                        {"actuator_id": 42, "position": 0.0, "velocity": 0.0},  # right_hip_roll_03
                        {"actuator_id": 44, "position": 0.0, "velocity": 0.0},  # right_knee_04
                        {"actuator_id": 45, "position": 0.0, "velocity": 0.0},  # right_ankle_02
                        # Left leg.
                        {"actuator_id": 31, "position": 0.0, "velocity": 0.0},  # left_hip_pitch_04
                        {"actuator_id": 32, "position": 0.0, "velocity": 0.0},  # left_hip_roll_03
                        {"actuator_id": 34, "position": 0.0, "velocity": 0.0},  # left_knee_04
                        {"actuator_id": 35, "position": 0.0, "velocity": 0.0},  # left_ankle_02
                    ]
                ),
                kos.imu.get_quaternion(),
            )

            # Gets the direction of gravity. The Z-axis is up.
            quat = R.from_quat([raw_quat.x, raw_quat.y, raw_quat.z, raw_quat.w])
            gravity_direction = quat.apply(np.array([0, 0, -1]))

            # Make the hips move in the opposite direction of gravity.
            scale = gravity_direction[0] + 0.05
            delta = scale * -300.0

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
