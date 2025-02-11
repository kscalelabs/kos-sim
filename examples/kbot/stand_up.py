"""Script for making the robot stand up from lying face-down on the ground."""

import argparse
import asyncio
import logging
from dataclasses import dataclass

import colorlogging
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


async def stand_up(kos: KOS) -> None:
    """Execute stand-up sequence from prone position."""
    # Phase 1: Push up with arms
    logger.info("Phase 1: Pushing up with arms")
    await kos.actuator.command_actuators(
        [
            # Arms push down
            {"actuator_id": 11, "position": -90.0},  # left shoulder pitch
            {"actuator_id": 14, "position": -90.0},  # left elbow
            {"actuator_id": 21, "position": -90.0},  # right shoulder pitch
            {"actuator_id": 24, "position": -90.0},  # right elbow
            # Tuck knees under
            {"actuator_id": 34, "position": 120.0},  # left knee
            {"actuator_id": 44, "position": 120.0},  # right knee
        ]
    )
    await asyncio.sleep(1.0)

    # Phase 2: Shift weight back and tuck legs
    logger.info("Phase 2: Shifting weight back")
    await kos.actuator.command_actuators(
        [
            # Arms maintain push
            {"actuator_id": 11, "position": -100.0},  # left shoulder pitch
            {"actuator_id": 14, "position": -90.0},  # left elbow
            {"actuator_id": 21, "position": -100.0},  # right shoulder pitch
            {"actuator_id": 24, "position": -90.0},  # right elbow
            # Hips and knees position for squat
            {"actuator_id": 31, "position": 90.0},  # left hip pitch
            {"actuator_id": 34, "position": 130.0},  # left knee
            {"actuator_id": 41, "position": 90.0},  # right hip pitch
            {"actuator_id": 44, "position": 130.0},  # right knee
            # Ankles prepare for standing
            {"actuator_id": 35, "position": -45.0},  # left ankle
            {"actuator_id": 45, "position": -45.0},  # right ankle
        ]
    )
    await asyncio.sleep(1.0)

    # Phase 3: Squat position
    logger.info("Phase 3: Moving to squat position")
    await kos.actuator.command_actuators(
        [
            # Arms support from front
            {"actuator_id": 11, "position": -45.0},  # left shoulder pitch
            {"actuator_id": 14, "position": -90.0},  # left elbow
            {"actuator_id": 21, "position": -45.0},  # right shoulder pitch
            {"actuator_id": 24, "position": -90.0},  # right elbow
            # Deep squat position
            {"actuator_id": 31, "position": 100.0},  # left hip pitch
            {"actuator_id": 34, "position": 120.0},  # left knee
            {"actuator_id": 41, "position": 100.0},  # right hip pitch
            {"actuator_id": 44, "position": 120.0},  # right knee
            # Ankles provide stability
            {"actuator_id": 35, "position": -30.0},  # left ankle
            {"actuator_id": 45, "position": -30.0},  # right ankle
        ]
    )
    await asyncio.sleep(1.0)

    # Phase 4: Stand up
    logger.info("Phase 4: Standing up")
    await kos.actuator.command_actuators(
        [
            # Arms at sides
            {"actuator_id": 11, "position": 0.0},  # left shoulder pitch
            {"actuator_id": 14, "position": 0.0},  # left elbow
            {"actuator_id": 21, "position": 0.0},  # right shoulder pitch
            {"actuator_id": 24, "position": 0.0},  # right elbow
            # Legs straighten
            {"actuator_id": 31, "position": 20.0},  # left hip pitch
            {"actuator_id": 34, "position": 40.0},  # left knee
            {"actuator_id": 41, "position": 20.0},  # right hip pitch
            {"actuator_id": 44, "position": 40.0},  # right knee
            # Ankles provide balance
            {"actuator_id": 35, "position": -20.0},  # left ankle
            {"actuator_id": 45, "position": -20.0},  # right ankle
        ]
    )
    await asyncio.sleep(1.0)


async def test_client(host: str = "localhost", port: int = 50051) -> None:
    logger.info("Starting stand-up client...")

    async with KOS(ip=host, port=port) as kos:
        # Start the robot lying down with zeroed joints.
        init_xyz = [0.0, 0.0, 0.2]
        init_quat = list(R.from_euler("xyz", [0.0, 90.0, 0.0], degrees=True).as_quat())
        init_joints = [0.0] * len(ACTUATOR_LIST)

        # Reset the simulation.
        await kos.sim.reset(initial_state={"qpos": init_xyz + init_quat + init_joints})

        # Configure all actuators
        for actuator in ACTUATOR_LIST:
            await kos.actuator.configure_actuator(
                actuator_id=actuator.actuator_id,
                kp=actuator.kp,
                kd=actuator.kd,
                max_torque=actuator.max_torque,
                torque_enabled=True,
            )

        # Execute stand-up sequence
        await stand_up(kos)
        logger.info("Stand-up sequence completed")


async def main() -> None:
    """Run the stand-up sequence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)
    await test_client(host=args.host, port=args.port)


if __name__ == "__main__":
    # python -m examples.kbot.stand_up
    asyncio.run(main())
