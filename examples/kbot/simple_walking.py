"""Run reinforcement learning on the robot simulator."""
import argparse
import asyncio
import logging
import math
import os
import subprocess
import time
from dataclasses import dataclass

import colorlogging
import numpy as np
import onnxruntime as ort
import pykos
from kinfer.inference.python import ONNXModel
from pykos import KOS
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


@dataclass
class Actuator:
    actuator_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


ACTUATOR_LIST: list[Actuator] = [
    Actuator(actuator_id=31, kp=300.0, kd=5.0, max_torque=60.0, joint_name="left_hip_pitch_04"),
    Actuator(actuator_id=32, kp=120.0, kd=5.0, max_torque=40.0, joint_name="left_hip_roll_03"),
    Actuator(actuator_id=33, kp=120.0, kd=5.0, max_torque=40.0, joint_name="left_hip_yaw_03"),
    Actuator(actuator_id=34, kp=300.0, kd=5.0, max_torque=60.0, joint_name="left_knee_04"),
    Actuator(actuator_id=35, kp=40.0, kd=5.0, max_torque=17.0, joint_name="left_ankle_02"),

    Actuator(actuator_id=41, kp=300.0, kd=5.0, max_torque=60.0, joint_name="right_hip_pitch_04"),
    Actuator(actuator_id=42, kp=120.0, kd=5.0, max_torque=40.0, joint_name="right_hip_roll_03"),
    Actuator(actuator_id=43, kp=120.0, kd=5.0, max_torque=40.0, joint_name="right_hip_yaw_03"),
    Actuator(actuator_id=44, kp=300.0, kd=5.0, max_torque=60.0, joint_name="right_knee_04"),
    Actuator(actuator_id=45, kp=40.0, kd=5.0, max_torque=17.0, joint_name="right_ankle_02"),  
]

ACTUATOR_IDS = [actuator.actuator_id for actuator in ACTUATOR_LIST]
ACTUATOR_ID_TO_POLICY_IDX = {
    31: 0,  # left_hip_pitch_04
    32: 1,  # left_hip_roll_03
    33: 2,  # left_hip_yaw_03
    34: 3,  # left_knee_04
    35: 4,  # left_ankle_02
    41: 5,  # right_hip_pitch_04
    42: 6,  # right_hip_roll_03
    43: 7,  # right_hip_yaw_03
    44: 8,  # right_knee_04
    45: 9,  # right_ankle_02
}

async def simple_walking(
    policy: ONNXModel,
    default_position: list[float], 
    host: str, 
    port: int
) -> None:
    async with KOS(ip=host, port=port) as sim_kos:
        await sim_kos.sim.reset(
            initial_state={
                    "qpos": [0.0, 0.0, 1.05, 0.0, 0.0, 0.0, 1.0] + default_position
                }
            )
        count = 0
        start_time = time.time()
        end_time = start_time + 10
        last_second = int(time.time())
        second_count = 0

        default = np.array(default_position)
        target_q = np.zeros(10, dtype=np.double)
        prev_actions = np.zeros(10, dtype=np.double)
        hist_obs = np.zeros(570, dtype=np.double)

        count_lowlevel = 0

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
        x_vel_cmd = 0.3
        y_vel_cmd = 0.0
        yaw_vel_cmd = 0.0
        frequency = 50

        start_time = time.time()
        while time.time() < end_time:
            loop_start_time = time.time()
        
            response, raw_quat = await asyncio.gather(
                sim_kos.actuator.get_actuators_state(ACTUATOR_IDS),
                sim_kos.imu.get_quaternion()
            )
            positions = np.array([math.radians(state.position) for state in response.states])
            velocities = np.array([math.radians(state.velocity) for state in response.states])
            r = R.from_quat([raw_quat.x, raw_quat.y, raw_quat.z, raw_quat.w])
            gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
            print(gvec)
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
            
            waiting_time = 1 / frequency
            loop_end_time = time.time()
            sleep_time = max(0, waiting_time - (loop_end_time - loop_start_time))
            await asyncio.sleep(sleep_time)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sim-only", action="store_true",
                        help="Run simulation only without connecting to the real robot")
    args = parser.parse_args()
 
    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    try:
        print("Running in simulation-only mode...")

        policy = ONNXModel("assets/simple_walking.onnx")
        default_position = [0.23, 0.0, 0.0, 0.441, -0.195, -0.23, 0.0, 0.0, -0.441, 0.195]

        sim_process = subprocess.Popen(["kos-sim", "kbot-v1", "--debug"])
        time.sleep(2)

        await simple_walking(policy, default_position, args.host, args.port)

    except Exception:
        logger.exception("Simulator error")
        raise
    finally:
        sim_process.terminate()
        sim_process.wait()


if __name__ == "__main__":
    asyncio.run(main())
