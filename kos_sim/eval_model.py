"""Real robot walking deployment script using PyKOS interface."""

import argparse
import asyncio
import math
import time
from copy import deepcopy
from typing import Dict, List, Tuple, Union
import csv

import numpy as np
import pygame
from kinfer.inference.python import ONNXModel
import pykos
from pykos import KOS
from scipy.spatial.transform import Rotation as R

ARM_IDS = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26]

upper_arm_ids = [11, 12, 13, 14, 15, 21, 22, 23, 24, 25]

# Ordered joint names for policy (legs only, top-down, left-right order)
JOINT_NAME_LIST = [
    "L_hip_y",    # Left leg, top to bottom
    "L_hip_x",
    "L_hip_z", 
    "L_knee",
    "L_ankle",
    "R_hip_y",    # Right leg, top to bottom
    "R_hip_x",
    "R_hip_z",
    "R_knee",
    "R_ankle"
]

# Joint mapping for KOS
JOINT_NAME_TO_ID = {
    # Left leg
    "L_hip_y": 31,
    "L_hip_x": 32,
    "L_hip_z": 33,
    "L_knee": 34,
    "L_ankle": 35,
    # Right leg
    "R_hip_y": 41,
    "R_hip_x": 42,
    "R_hip_z": 43,
    "R_knee": 44,
    "R_ankle": 45
}

# Joint signs for correct motion direction
JOINT_SIGNS = {
    # Left leg
    "L_hip_y": 1,
    "L_hip_x": 1,
    "L_hip_z": 1,
    "L_knee": -1,
    "L_ankle": 1,
    # Right leg
    "R_hip_y": 1,
    "R_hip_x": 1,
    "R_hip_z": 1,
    "R_knee": 1,
    "R_ankle": -1
}

def handle_keyboard_input() -> None:
    """Handle keyboard input for velocity commands."""
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
    
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        x_vel_cmd += 0.0005
    if keys[pygame.K_DOWN]:
        x_vel_cmd -= 0.0005
    if keys[pygame.K_LEFT]:
        y_vel_cmd += 0.0005
    if keys[pygame.K_RIGHT]:
        y_vel_cmd -= 0.0005
    if keys[pygame.K_a]:
        yaw_vel_cmd += 0.001
    if keys[pygame.K_z]:
        yaw_vel_cmd -= 0.001

class RobotState:
    """Tracks robot state and handles offsets."""
    def __init__(self, joint_names: List[str], joint_signs: Dict[str, float]):
        self.joint_offsets = {name: 0.0 for name in joint_names}
        self.joint_signs = joint_signs
        self.orn_offset = None

    async def offset_in_place(self, kos: KOS, joint_names: List[str]) -> None:
        """Capture current position as zero offset."""
        # Get current joint positions (in degrees)
        states = await kos.actuator.get_actuators_state([JOINT_NAME_TO_ID[name] for name in joint_names])
        current_positions = {name: states.states[i].position for i, name in enumerate(joint_names)}
        
        # Store negative of current positions as offsets (in degrees)
        # HACK: No offsets for now
        self.joint_offsets = {name: 0.0 for name, _ in current_positions.items()}#-pos for name, pos in current_positions.items()}

        # Store IMU offset
        imu_data = await kos.imu.get_euler_angles()
        initial_quat = R.from_euler('xyz', [imu_data.roll, imu_data.pitch, imu_data.yaw], degrees=True).as_quat()
        self.orn_offset = R.from_quat(initial_quat).inv()

    async def get_obs(self, kos: KOS) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get robot state with offset compensation."""
        # Batch state requests
        states, imu_data = await asyncio.gather(
            kos.actuator.get_actuators_state([JOINT_NAME_TO_ID[name] for name in JOINT_NAME_LIST]),
            kos.imu.get_euler_angles()
        )
        
        # Apply offsets and signs to positions and convert to radians
        q = np.array([
            np.deg2rad((states.states[i].position + self.joint_offsets[name]) * self.joint_signs[name])
            for i, name in enumerate(JOINT_NAME_LIST)
        ], dtype=np.float32)
        
        # Apply signs to velocities and convert to radians
        dq = np.array([
            np.deg2rad(states.states[i].velocity * self.joint_signs[name])
            for i, name in enumerate(JOINT_NAME_LIST)
        ], dtype=np.float32)

        # Process IMU data with offset compensation
        current_quat = R.from_euler('xyz', [imu_data.roll, imu_data.pitch, imu_data.yaw], degrees=True).as_quat()
        if self.orn_offset is not None:
            # Apply the offset by quaternion multiplication
            current_rot = R.from_quat(current_quat)
            quat = (self.orn_offset * current_rot).as_quat()
        else:
            quat = current_quat

        # Calculate gravity vector with offset compensation
        r = R.from_quat(quat)
        gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True)

        return q, dq, quat, gvec

    def apply_command(self, position: float, joint_name: str) -> float:
        """Apply sign first, then offset to outgoing command. Convert from radians to degrees."""
        # Convert from radians to degrees since position from policy is in radians
        position_deg = np.rad2deg(position)
        return position_deg * self.joint_signs[joint_name] - self.joint_offsets[joint_name]

async def run_robot(
    kos: KOS,
    policy: ONNXModel,
    model_info: Dict[str, Union[float, List[float], str]],
    keyboard_use: bool = False,
    duration: float = 60.0,
) -> None:
    """Run the walking policy on the real robot."""
    
    # Initialize process time tracking
    process_times = []
    
    # Initialize robot state handler
    robot_state = RobotState(JOINT_NAME_LIST, JOINT_SIGNS)

    sim = kos.sim
    initial_state = {"qpos": [0.0, 0.0, 1.16620985, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    print()
    await sim.reset(initial_state=initial_state)
    await sim.set_paused(True)
    # Open CSV files for logging
    with open('inputs.csv', 'w', newline='') as inputs_file, \
         open('outputs.csv', 'w', newline='') as outputs_file:
        
        # Initialize CSV writers
        inputs_writer = csv.writer(inputs_file)
        outputs_writer = csv.writer(outputs_file)
        
        # Write headers
        inputs_writer.writerow([
            'timestamp', 'x_vel', 'y_vel', 'yaw_vel', 
            *[f'q_{i}' for i in range(10)],  # joint positions
            *[f'dq_{i}' for i in range(10)],  # joint velocities
            *['quat_w', 'quat_x', 'quat_y', 'quat_z'],  # quaternion
            *['gvec_x', 'gvec_y', 'gvec_z']  # gravity vector
        ])
        
        outputs_writer.writerow([
            'timestamp', *[f'target_q_{i}' for i in range(10)]
        ])

        # Configure motors
        # print("Configuring motors...")
        leg_ids = [JOINT_NAME_TO_ID[name] for name in JOINT_NAME_LIST]
        for joint_id in leg_ids:
            await kos.actuator.configure_actuator(actuator_id=joint_id, torque_enabled=True)
        
        # Freeze upper arms in place
        arm_commands = []
        for arm_id in upper_arm_ids:
            arm_commands.append({"actuator_id": arm_id, "position": 0.0})
        await kos.actuator.command_actuators(arm_commands)

        # Capture current position as zero
        # print("Capturing current position as zero...")
        await robot_state.offset_in_place(kos, JOINT_NAME_LIST)

        # Initialize policy state
        default = np.array(model_info["default_standing"])
        target_q = np.zeros(model_info["num_actions"], dtype=np.float32)
        prev_actions = np.zeros(model_info["num_actions"], dtype=np.float32)
        hist_obs = np.zeros(model_info["num_observations"], dtype=np.float32)
        count_policy = 0

        print(f"Going to zero position...")
        await kos.actuator.command_actuators([{"actuator_id": joint_id, "position": 0.0} for joint_id in leg_ids])

        # await kos.actuator.command_actuators([{"actuator_id": 11, "position": 0.0}])
        # return

        for i in range(3, -1, -1):
            print(f"Starting in {i} seconds...")
            await asyncio.sleep(1)

        try:
            await sim.set_paused(False)
            while True:
                process_start = time.time()
                if keyboard_use:
                    handle_keyboard_input()

                try:
                    # Get robot state with offset compensation
                    q, dq, quat, gvec = await robot_state.get_obs(kos)

                    # Log inputs
                    inputs_writer.writerow([
                        time.time(),
                        x_vel_cmd, y_vel_cmd, yaw_vel_cmd,
                        *q,  # joint positions
                        *dq,  # joint velocities
                        *quat,  # quaternion
                        *gvec  # gravity vector
                    ])
                    inputs_file.flush()  # Ensure data is written immediately

                    # Prepare policy inputs and run policy
                    input_data = {
                        "x_vel.1": np.array([x_vel_cmd], dtype=np.float32),
                        "y_vel.1": np.array([y_vel_cmd], dtype=np.float32),
                        "rot.1": np.array([yaw_vel_cmd], dtype=np.float32),
                        "t.1": np.array([count_policy * model_info["policy_dt"]], dtype=np.float32),
                        "dof_pos.1": (q - default).astype(np.float32),
                        "dof_vel.1": dq.astype(np.float32),
                        "prev_actions.1": prev_actions.astype(np.float32),
                        "projected_gravity.1": gvec.astype(np.float32),
                        "buffer.1": hist_obs.astype(np.float32),
                    }

                    # Run policy
                    policy_output = policy(input_data)
                    target_q = policy_output["actions_scaled"]
                    prev_actions = policy_output["actions"]
                    hist_obs = policy_output["x.3"]

                    # Log outputs
                    outputs_writer.writerow([time.time(), *target_q])
                    outputs_file.flush()  # Ensure data is written immediately

                    # Apply commands with offset compensation
                    commands = []
                    for i, joint_name in enumerate(JOINT_NAME_LIST):
                        joint_id = JOINT_NAME_TO_ID[joint_name]
                        position = robot_state.apply_command(float(target_q[i] + default[i]), joint_name)
                        commands.append({"actuator_id": joint_id, "position": position})
                    await kos.actuator.command_actuators(commands)

                    process_time = time.time() - process_start
                    process_times.append(process_time)
                    
                    sleep = model_info["policy_dt"] - process_time
                    if sleep > 0:
                        await asyncio.sleep(sleep)
                    # else:
                    #     print(f"Policy took {process_time:.4f} seconds")

                    count_policy += 1

                except asyncio.CancelledError:
                    raise

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nStopping walking...")
            if process_times:
                avg_time = sum(process_times) / len(process_times)
                max_time = max(process_times)
                print(f"\nProcess Time Statistics:")
                print(f"Average: {avg_time:.4f} seconds")
                print(f"Median: {np.median(process_times):.4f} seconds")
                print(f"Maximum: {max_time:.4f} seconds")
                print(f"Num too slow: {len([t for t in process_times if t > 0.02])}")
                print(f"Percentage too slow: {len([t for t in process_times if t > 0.02]) / len(process_times):.4f}")
                print(f"Total Iterations: {len(process_times)}")
        finally:
            # Disable torque on exit
            for joint_id in leg_ids + upper_arm_ids:
                await kos.actuator.configure_actuator(actuator_id=joint_id, torque_enabled=False)

async def main():
    parser = argparse.ArgumentParser(description="Real robot walking deployment script.")
    parser.add_argument("--load_model", type=str, required=True, help="Path to policy model")
    parser.add_argument("--keyboard_use", action="store_true", help="Enable keyboard control")
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    parser.add_argument("--port", type=int, default=50051, help="Robot port")
    args = parser.parse_args()

    # Initialize KOS and policy
    async with KOS(ip=args.ip, port=args.port) as kos:
        policy = ONNXModel(args.load_model)
            
        # Get model info from policy metadata
        metadata = policy.get_metadata()
        print(metadata)
        model_info = {
            "num_actions": metadata["num_actions"],
            "num_observations": metadata["num_observations"],
            "robot_effort": metadata["robot_effort"],
            "robot_stiffness": metadata["robot_stiffness"],
            "robot_damping": metadata["robot_damping"],
            "tau_factor": metadata["tau_factor"],
            "policy_dt": metadata["sim_dt"] * metadata["sim_decimation"],
            "default_standing": metadata["default_standing"],
            # "policy_dt": 1,
        }

        print(f"Model Info: {model_info}")

        # Initialize velocity commands
        global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
        if args.keyboard_use:
            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
            pygame.init()
            pygame.display.set_caption("Robot Control")
        else:
            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0

        # Run robot control
        await run_robot(
            kos=kos,
            policy=policy,
            model_info=model_info,
            keyboard_use=args.keyboard_use,
        )

if __name__ == "__main__":
    asyncio.run(main())