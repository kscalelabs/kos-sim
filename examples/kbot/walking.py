"""Interactive example script for a simple walking gait."""

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto

import colorlogging
import numpy as np
from pykos import KOS
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class WalkingPhase(Enum):
    """Walking gait phases."""

    LEFT_STANCE = auto()  # Left foot on ground, right foot moving forward
    RIGHT_STANCE = auto()  # Right foot on ground, left foot moving forward


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
    logger.info("Starting walking client...")

    async with KOS(ip=host, port=port) as kos:
        # Reset the simulation
        await kos.sim.reset()

        # Configure all actuators
        for actuator in ACTUATOR_LIST:
            await kos.actuator.configure_actuator(
                actuator_id=actuator.actuator_id,
                kp=actuator.kp,
                kd=actuator.kd,
                max_torque=actuator.max_torque,
                torque_enabled=True,
            )

        # Initialize walking parameters
        phase = WalkingPhase.LEFT_STANCE
        phase_duration = 2.0  # Slower steps for better stability
        start_time = time.time()
        next_time = start_time + 1 / 50
        step_height = 8.0  # Lower step height
        step_length = 8.0  # Shorter steps
        hip_swing = 5.0  # Reduced hip swing

        # Start in a wider, more stable stance
        await kos.actuator.command_actuators(
            [
                # Right leg - initial stance
                {"actuator_id": 41, "position": -20.0},  # right_hip_pitch
                {"actuator_id": 42, "position": -5.0},  # right_hip_roll - wider stance
                {"actuator_id": 44, "position": -40.0},  # right_knee - less bent
                {"actuator_id": 45, "position": 20.0},  # right_ankle
                # Left leg - initial stance
                {"actuator_id": 31, "position": 20.0},  # left_hip_pitch
                {"actuator_id": 32, "position": 5.0},  # left_hip_roll - wider stance
                {"actuator_id": 34, "position": 40.0},  # left_knee - less bent
                {"actuator_id": 35, "position": -20.0},  # left_ankle
            ]
        )

        # Wait longer for initial pose to stabilize
        await asyncio.sleep(2.0)
        start_time = time.time()

        while True:
            current_time = time.time()
            phase_time = (current_time - start_time) % phase_duration
            phase_progress = phase_time / phase_duration

            # Switch phases
            if phase_time < 0.01:
                phase = WalkingPhase.RIGHT_STANCE if phase == WalkingPhase.LEFT_STANCE else WalkingPhase.LEFT_STANCE
                logger.info("Switching to phase: %s", phase)

            # Get IMU data for balance
            raw_quat = await kos.imu.get_quaternion()
            quat = R.from_quat([raw_quat.x, raw_quat.y, raw_quat.z, raw_quat.w])
            gravity_direction = quat.apply(np.array([0, 0, -1]))

            # Increased balance correction gains
            lateral_correction = gravity_direction[0] * -25.0
            sagittal_correction = gravity_direction[1] * -25.0

            # Modified motion profiles for more pronounced lift
            lift_profile = np.sin(np.pi * phase_progress) ** 2  # Sharper lift
            forward_profile = 0.5 * (1 - np.cos(2 * np.pi * phase_progress))

            if phase == WalkingPhase.LEFT_STANCE:
                right_lift = step_height * lift_profile
                right_forward = step_length * (forward_profile - 0.5)
                left_forward = -step_length * (0.5 - forward_profile)

                commands = [
                    # Stance (left) leg
                    {"actuator_id": 31, "position": 20.0 + left_forward + sagittal_correction},
                    {"actuator_id": 32, "position": 5.0 + lateral_correction},
                    {"actuator_id": 34, "position": 40.0},
                    {"actuator_id": 35, "position": -20.0 - sagittal_correction},
                    # Swing (right) leg - increased lift motion
                    {"actuator_id": 41, "position": -20.0 + right_forward + hip_swing * lift_profile},
                    {"actuator_id": 42, "position": -5.0 + lateral_correction},
                    {"actuator_id": 44, "position": -40.0 - right_lift * 1.5},  # Increased knee bend
                    {"actuator_id": 45, "position": 20.0 + right_lift * 0.5},  # Ankle compensation
                ]
            else:
                left_lift = step_height * lift_profile
                left_forward = step_length * (forward_profile - 0.5)
                right_forward = -step_length * (0.5 - forward_profile)

                commands = [
                    # Swing (left) leg - increased lift motion
                    {"actuator_id": 31, "position": 20.0 + left_forward + hip_swing * lift_profile},
                    {"actuator_id": 32, "position": 5.0 + lateral_correction},
                    {"actuator_id": 34, "position": 40.0 - left_lift * 1.5},  # Increased knee bend
                    {"actuator_id": 35, "position": -20.0 + left_lift * 0.5},  # Ankle compensation
                    # Stance (right) leg
                    {"actuator_id": 41, "position": -20.0 + right_forward + sagittal_correction},
                    {"actuator_id": 42, "position": -5.0 + lateral_correction},
                    {"actuator_id": 44, "position": -40.0},
                    {"actuator_id": 45, "position": 20.0 - sagittal_correction},
                ]

            await kos.actuator.command_actuators(commands)

            # Maintain control frequency
            if next_time > current_time:
                await asyncio.sleep(next_time - current_time)
            next_time += 1 / 50


async def main() -> None:
    """Runs the main walking control loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)
    await test_client(host=args.host, port=args.port)


if __name__ == "__main__":
    # python -m examples.kbot.walking
    asyncio.run(main())
