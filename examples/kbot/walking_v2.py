"""Run reinforcement learning on the robot simulator."""

import argparse
import asyncio
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import colorlogging
import numpy as np
import onnxruntime as ort
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
    joint_name: str


ACTUATOR_LIST: list[Actuator] = [
    Actuator(actuator_id=31, nn_id=0, kp=250.0, kd=5.0, max_torque=40.0, joint_name="left_hip_pitch_04"),
    Actuator(actuator_id=32, nn_id=1, kp=120.0, kd=5.0, max_torque=30.0, joint_name="left_hip_roll_03"),
    Actuator(actuator_id=33, nn_id=2, kp=120.0, kd=5.0, max_torque=30.0, joint_name="left_hip_yaw_03"),
    Actuator(actuator_id=34, nn_id=3, kp=250.0, kd=5.0, max_torque=40.0, joint_name="left_knee_04"),
    Actuator(actuator_id=35, nn_id=4, kp=40.0, kd=5.0, max_torque=10.0, joint_name="left_ankle_02"),
    Actuator(actuator_id=41, nn_id=5, kp=250.0, kd=5.0, max_torque=40.0, joint_name="right_hip_pitch_04"),
    Actuator(actuator_id=42, nn_id=6, kp=120.0, kd=5.0, max_torque=30.0, joint_name="right_hip_roll_03"),
    Actuator(actuator_id=43, nn_id=7, kp=120.0, kd=5.0, max_torque=30.0, joint_name="right_hip_yaw_03"),
    Actuator(actuator_id=44, nn_id=8, kp=250.0, kd=5.0, max_torque=40.0, joint_name="right_knee_04"),
    Actuator(actuator_id=45, nn_id=9, kp=40.0, kd=5.0, max_torque=10.0, joint_name="right_ankle_02"),
]

ACTUATOR_ID_TO_POLICY_IDX = {actuator.actuator_id: actuator.nn_id for actuator in ACTUATOR_LIST}

ACTUATOR_IDS = [actuator.actuator_id for actuator in ACTUATOR_LIST]


async def simple_walking(
    model_path: str | Path,
    default_position: list[float],
    host: str,
    port: int,
    num_seconds: float | None = 10.0,
) -> None:
    """Runs a simple walking policy.

    Args:
        model_path: The path to the ONNX model.
        default_position: The default joint positions for the legs.
        host: The host to connect to.
        port: The port to connect to.
        num_seconds: The number of seconds to run the policy for.
    """
    assert len(default_position) == len(ACTUATOR_LIST)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    session = ort.InferenceSession(model_path)

    # Get input and output details
    output_details = [{"name": x.name, "shape": x.shape, "type": x.type} for x in session.get_outputs()]

    def policy(input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        results = session.run(None, input_data)
        return {output_details[i]["name"]: results[i] for i in range(len(output_details))}

    async with KOS(ip=host, port=port) as sim_kos:
        for actuator in ACTUATOR_LIST:
            await sim_kos.actuator.configure_actuator(
                actuator_id=actuator.actuator_id,
                kp=actuator.kp,
                kd=actuator.kd,
                max_torque=actuator.max_torque,
            )

        await sim_kos.sim.reset(
            pos={"x": 0.0, "y": 0.0, "z": 0.795},
            quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            joints=[
                {
                    "name": actuator.joint_name,
                    "pos": pos,
                }
                for actuator, pos in zip(ACTUATOR_LIST, default_position)
            ],
        )
        start_time = time.time()
        end_time = None if num_seconds is None else start_time + num_seconds

        default = np.array(default_position)
        target_q = np.zeros(10, dtype=np.double)
        prev_actions = np.zeros(10, dtype=np.double)
        hist_obs = np.zeros(570, dtype=np.double)

        input_data = {
            "x_vel.1": np.zeros(1).astype(np.float32),
            "y_vel.1": np.zeros(1).astype(np.float32),
            "rot.1": np.zeros(1).astype(np.float32),
            "t.1": np.zeros(1).astype(np.float32),
            "dof_pos.1": np.zeros(10).astype(np.float32),
            "dof_vel.1": np.zeros(10).astype(np.float32),
            "prev_actions.1": np.zeros(10).astype(np.float32),
            "projected_gravity.1": np.zeros(3).astype(np.float32),
            "buffer.1": np.zeros(570).astype(np.float32),
        }

        x_vel_cmd = 0.5
        y_vel_cmd = 0.0
        yaw_vel_cmd = 0.0
        frequency = 50

        start_time = time.time()
        next_time = start_time + 1 / frequency

        while end_time is None or time.time() < end_time:
            response, raw_quat = await asyncio.gather(
                sim_kos.actuator.get_actuators_state(ACTUATOR_IDS),
                sim_kos.imu.get_quaternion(),
            )
            positions = np.array([math.radians(state.position) for state in response.states])
            velocities = np.array([math.radians(state.velocity) for state in response.states])
            r = R.from_quat([raw_quat.x, raw_quat.y, raw_quat.z, raw_quat.w])

            gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)

            # Need to apply a transformation from the IMU frame to the frame
            # that we used to train the original model.
            gvec = np.array([-gvec[2], -gvec[1], gvec[0]])

            cur_pos_obs = positions - default
            cur_vel_obs = velocities
            input_data["x_vel.1"] = np.array([x_vel_cmd], dtype=np.float32)
            input_data["y_vel.1"] = np.array([y_vel_cmd], dtype=np.float32)
            input_data["rot.1"] = np.array([yaw_vel_cmd], dtype=np.float32)
            input_data["t.1"] = np.array([time.time() - start_time], dtype=np.float32)
            input_data["dof_pos.1"] = cur_pos_obs.astype(np.float32)
            input_data["dof_vel.1"] = cur_vel_obs.astype(np.float32)
            input_data["prev_actions.1"] = prev_actions.astype(np.float32)
            input_data["projected_gravity.1"] = gvec.astype(np.float32)
            input_data["buffer.1"] = hist_obs.astype(np.float32)

            policy_output = policy(input_data)
            positions = policy_output["actions_scaled"]
            curr_actions = policy_output["actions"]
            hist_obs = policy_output["x.3"]
            prev_actions = curr_actions

            target_q = positions + default

            commands = []
            for actuator_id in ACTUATOR_IDS:
                policy_idx = ACTUATOR_ID_TO_POLICY_IDX[actuator_id]
                raw_value = target_q[policy_idx]
                command_deg = raw_value
                command_deg = math.degrees(raw_value)
                commands.append({"actuator_id": actuator_id, "position": command_deg})

            await sim_kos.actuator.command_actuators(commands)
            await asyncio.sleep(max(0, next_time - time.time()))
            next_time += 1 / frequency


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--num-seconds", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", type=str, default="simple_walking_v2.onnx")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    model_path = Path(__file__).parent / args.model

    # Defines the default joint positions for the legs.
    default_position = [0.23, 0.0, 0.0, 0.441, -0.195, -0.23, 0.0, 0.0, -0.441, 0.195]
    await simple_walking(model_path, default_position, args.host, args.port, args.num_seconds)


if __name__ == "__main__":
    asyncio.run(main())
