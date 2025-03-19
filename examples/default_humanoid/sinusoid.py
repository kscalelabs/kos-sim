"""Adhoc testing script."""

import asyncio
import math
import time
from dataclasses import dataclass

from pykos import KOS


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


ACTUATOR_LIST: list[Actuator] = [
    Actuator(actuator_id=1, nn_id=0, kp=50.0, kd=1.0, max_torque=30.0, joint_name="abdomen_y"),
    Actuator(actuator_id=2, nn_id=1, kp=50.0, kd=1.0, max_torque=30.0, joint_name="abdomen_z"),
    Actuator(actuator_id=3, nn_id=2, kp=50.0, kd=1.0, max_torque=30.0, joint_name="abdomen_x"),
    Actuator(actuator_id=4, nn_id=3, kp=50.0, kd=1.0, max_torque=30.0, joint_name="hip_x_right"),
    Actuator(actuator_id=5, nn_id=4, kp=50.0, kd=1.0, max_torque=30.0, joint_name="hip_z_right"),
    Actuator(actuator_id=6, nn_id=5, kp=50.0, kd=1.0, max_torque=30.0, joint_name="hip_y_right"),
    Actuator(actuator_id=7, nn_id=6, kp=50.0, kd=1.0, max_torque=30.0, joint_name="knee_right"),
    Actuator(actuator_id=8, nn_id=7, kp=50.0, kd=1.0, max_torque=30.0, joint_name="ankle_x_right"),
    Actuator(actuator_id=9, nn_id=8, kp=50.0, kd=1.0, max_torque=30.0, joint_name="ankle_y_right"),
    Actuator(actuator_id=10, nn_id=9, kp=50.0, kd=1.0, max_torque=30.0, joint_name="hip_x_left"),
    Actuator(actuator_id=11, nn_id=10, kp=50.0, kd=1.0, max_torque=30.0, joint_name="hip_z_left"),
    Actuator(actuator_id=12, nn_id=11, kp=50.0, kd=1.0, max_torque=30.0, joint_name="hip_y_left"),
    Actuator(actuator_id=13, nn_id=12, kp=50.0, kd=1.0, max_torque=30.0, joint_name="knee_left"),
    Actuator(actuator_id=14, nn_id=13, kp=50.0, kd=1.0, max_torque=30.0, joint_name="ankle_x_left"),
    Actuator(actuator_id=15, nn_id=14, kp=50.0, kd=1.0, max_torque=30.0, joint_name="ankle_y_left"),
    Actuator(actuator_id=16, nn_id=15, kp=50.0, kd=1.0, max_torque=30.0, joint_name="shoulder1_right"),
    Actuator(actuator_id=17, nn_id=16, kp=50.0, kd=1.0, max_torque=30.0, joint_name="shoulder2_right"),
    Actuator(actuator_id=18, nn_id=17, kp=50.0, kd=1.0, max_torque=30.0, joint_name="elbow_right"),
    Actuator(actuator_id=19, nn_id=18, kp=50.0, kd=1.0, max_torque=30.0, joint_name="shoulder1_left"),
    Actuator(actuator_id=20, nn_id=19, kp=50.0, kd=1.0, max_torque=30.0, joint_name="shoulder2_left"),
    Actuator(actuator_id=21, nn_id=20, kp=50.0, kd=1.0, max_torque=30.0, joint_name="elbow_left"),
]

async def configure_joints(kos: KOS) -> None:
    for actuator in ACTUATOR_LIST:
        await kos.actuator.configure_actuator(
            actuator_id=actuator.actuator_id,
            kp=actuator.kp,
            kd=actuator.kd,
            max_torque=actuator.max_torque,
        )

async def reset_joints(kos: KOS) -> None:
    await kos.sim.reset(
        pos={"x": 0.0, "y": 0.0, "z": 2.05},
        quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        joints=[
            {"name": actuator.joint_name, "pos": pos}
            for actuator, pos in zip(ACTUATOR_LIST, [0.0] * len(ACTUATOR_LIST))
        ],
    )

async def command_zero_joints(kos: KOS) -> None:
    await kos.actuator.command_actuators(
        [{"actuator_id": actuator.actuator_id, "position": 0.0, "velocity": 0.0} for actuator in ACTUATOR_LIST]
    )

async def main() -> None:
    async with KOS() as kos:

        await reset_joints(kos)
        await configure_joints(kos)

        while True:
            # Sinusoid.
            t = time.time()
            amplitude = math.sin(t * 2 * math.pi * 0.1) * 10.0
            pos = math.sin(t * 2 * math.pi) * amplitude
            vel = math.cos(t * 2 * math.pi) * amplitude * 2 * math.pi

            await kos.actuator.command_actuators(
                [
                    {
                        "actuator_id": 14,
                        "position": pos,
                        "velocity": vel,
                    },
                    {
                        "actuator_id": 21,
                        "position": -pos,
                        "velocity": -vel,
                    },
                ]
            )

            cur_pos = await kos.actuator.get_actuators_state([14, 21])
            print(f"pos: {cur_pos}")

            await asyncio.sleep(0.02)

# Start the server with
# `python -m kos_sim.server default-humanoid`
if __name__ == "__main__":
    # `python -m examples.default_humanoid.sinusoid`
    asyncio.run(main())
