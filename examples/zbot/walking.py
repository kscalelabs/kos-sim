"""Interactive example script for walking in the KOS simulator."""

import argparse
import asyncio
import logging
import time
import math
import colorlogging
import numpy as np
from pykos import KOS

logger = logging.getLogger(__name__)

ACTUATOR_MAPPING = {
    # Left arm
    "left_shoulder_yaw": 11,
    "left_shoulder_pitch": 12,
    "left_elbow": 13,
    "left_gripper": 14,
    # Right arm
    "right_shoulder_yaw": 21,
    "right_shoulder_pitch": 22,
    "right_elbow": 23,
    "right_gripper": 24,
    # Left leg
    "left_hip_yaw": 31,
    "left_hip_roll": 32,
    "left_hip_pitch": 33,
    "left_knee": 34,
    "left_ankle": 35,
    # Right leg
    "right_hip_yaw": 41,
    "right_hip_roll": 42,
    "right_hip_pitch": 43,
    "right_knee": 44,
    "right_ankle": 45,
}


class BipedController:
    """
    Advanced bipedal walking controller with sophisticated balance and gait control.
    """

    def __init__(self, lateral_movement_enabled=True):
        # Added parameter to control lateral movements
        self.lateral_movement_enabled = lateral_movement_enabled

        self.roll_offset = math.radians(0)
        self.hip_pitch_offset = math.radians(15)  # Reduced from 20 to 15 degrees

        # -----------
        # Gait params - Adjusted for simulation stability
        # -----------
        self.LEG_LENGTH = 180.0  # mm
        self.hip_forward_offset = 2.04
        self.nominal_leg_height = 165.0  # Slightly lower stance
        self.initial_leg_height = 170.0  # Reduced initial height
        self.gait_phase = 0
        self.walking_enabled = True

        # -----------
        # Variables for cyclical stepping - More conservative values
        # -----------
        self.stance_foot_index = 0  # 0 or 1
        self.step_cycle_length = 8  # Increased from 4 to 8 for slower, more stable steps
        self.step_cycle_counter = 0
        self.lateral_foot_shift = 8  # Reduced from 12 for less aggressive lateral movement
        self.max_foot_lift = 8  # Reduced from 10 for lower steps
        self.double_support_fraction = 0.3  # Increased from 0.2 for more stable transitions
        self.current_foot_lift = 0.0

        # Add a base lateral offset to widen the stance
        self.base_stance_width = 3.0  # Increased from 2.0 for wider, more stable stance

        self.forward_offset = [0.0, 0.0]
        self.accumulated_forward_offset = 0.0
        self.previous_stance_foot_offset = 0.0
        self.previous_swing_foot_offset = 0.0
        self.step_length = 10.0  # Reduced from 15.0 for shorter, more stable steps

        self.lateral_offset = 0.0
        self.dyi = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        # The joint angle arrays
        self.K0 = [0.0, 0.0]  # hip pitch
        self.K1 = [0.0, 0.0]  # hip roll
        self.H = [0.0, 0.0]  # knee
        self.A0 = [0.0, 0.0]  # ankle pitch

    def control_foot_position(self, x, y, h, side):
        """
        Compute joint angles given the desired foot position (x, y, h).
        """
        k = math.sqrt(x * x + (y * y + h * h))
        k = min(k, self.LEG_LENGTH)  # Ensure k does not exceed LEG_LENGTH

        if abs(k) < 1e-8:
            alpha = 0.0
        else:
            alpha = math.asin(x / k)

        cval = max(min(k / self.LEG_LENGTH, 1.0), -1.0)
        gamma = math.acos(cval)

        self.K0[side] = gamma + alpha  # hip pitch
        self.H[side] = 2.0 * gamma + 0.3  # knee, increased pitch by adding 0.3 radians
        ankle_pitch_offset = 0.3  # Increased from 0.2 to 0.4 radians to compensate for forward lean

        self.A0[side] = gamma - alpha + ankle_pitch_offset  # ankle pitch with offset

        hip_roll = math.atan2(y, h) if abs(h) >= 1e-8 else 0.0
        self.K1[side] = hip_roll + self.roll_offset

    def virtual_balance_adjustment(self):
        """
        Compute a virtual center-of-mass (CoM) based on the current foot positions,
        and use a simple proportional feedback to adjust the lateral offset.
        """
        left_foot_x = self.forward_offset[0] - self.hip_forward_offset
        left_foot_y = -self.lateral_offset + 1.0
        right_foot_x = self.forward_offset[1] - self.hip_forward_offset
        right_foot_y = self.lateral_offset + 1.0

        # Compute estimated CoM as the average of the feet positions:
        com_x = (left_foot_x + right_foot_x) / 2.0
        com_y = (left_foot_y + right_foot_y) / 2.0

        # Desired CoM lateral position
        desired_com_y = 1.0

        # Compute the lateral error
        error_y = desired_com_y - com_y

        # Apply a proportional gain to adjust lateral offset
        feedback_gain = 0.1
        adjustment = feedback_gain * error_y

        # Update the lateral_offset value to help "steer" the CoM back toward desired position
        self.lateral_offset += adjustment

    def update_gait(self):
        """
        Update the internal state machine and foot positions each timestep.
        Now incorporates a virtual balance adjustment loop using the kinematic model.
        """
        if self.gait_phase == 0:
            # Ramping down from initial_leg_height to nominal_leg_height
            if self.initial_leg_height > self.nominal_leg_height + 0.1:
                self.initial_leg_height -= 1.0
            else:
                self.gait_phase = 10

            # Keep both feet together
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.initial_leg_height, 0)
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.initial_leg_height, 1)

        elif self.gait_phase == 10:
            # Idle
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.nominal_leg_height, 0)
            self.control_foot_position(-self.hip_forward_offset, 0.0, self.nominal_leg_height, 1)
            if self.walking_enabled:
                self.step_length = 20.0
                self.gait_phase = 20

        elif self.gait_phase in [20, 30]:
            # Precompute values used multiple times
            sin_value = math.sin(math.pi * self.step_cycle_counter / self.step_cycle_length)
            half_cycle = self.step_cycle_length / 2.0

            if self.lateral_movement_enabled:
                lateral_shift = self.lateral_foot_shift * sin_value
                self.lateral_offset = lateral_shift if self.stance_foot_index == 0 else -lateral_shift
                self.virtual_balance_adjustment()
            else:
                self.lateral_offset = 0.0

            if self.step_cycle_counter < half_cycle:
                fraction = self.step_cycle_counter / self.step_cycle_length
                self.forward_offset[self.stance_foot_index] = self.previous_stance_foot_offset * (1.0 - 2.0 * fraction)
            else:
                fraction = 2.0 * self.step_cycle_counter / self.step_cycle_length - 1.0
                self.forward_offset[self.stance_foot_index] = (
                    -(self.step_length - self.accumulated_forward_offset) * fraction
                )

            if self.gait_phase == 20:
                if self.step_cycle_counter < (self.double_support_fraction * self.step_cycle_length):
                    self.forward_offset[self.stance_foot_index ^ 1] = self.previous_swing_foot_offset - (
                        self.previous_stance_foot_offset - self.forward_offset[self.stance_foot_index]
                    )
                else:
                    self.previous_swing_foot_offset = self.forward_offset[self.stance_foot_index ^ 1]
                    self.gait_phase = 30

            if self.gait_phase == 30:
                start_swing = int(self.double_support_fraction * self.step_cycle_length)
                denom = (1.0 - self.double_support_fraction) * self.step_cycle_length
                if denom < 1e-8:
                    denom = 1.0
                frac = (-math.cos(math.pi * (self.step_cycle_counter - start_swing) / denom) + 1.0) / 2.0
                self.forward_offset[self.stance_foot_index ^ 1] = self.previous_swing_foot_offset + frac * (
                    self.step_length - self.accumulated_forward_offset - self.previous_swing_foot_offset
                )

            i = int(self.double_support_fraction * self.step_cycle_length)
            if self.step_cycle_counter > i:
                self.current_foot_lift = self.max_foot_lift * math.sin(
                    math.pi * (self.step_cycle_counter - i) / (self.step_cycle_length - i)
                )
            else:
                self.current_foot_lift = 0.0

            if self.stance_foot_index == 0:
                # left foot = stance
                self.control_foot_position(
                    self.forward_offset[0] - self.hip_forward_offset,
                    -self.lateral_offset - self.base_stance_width,
                    self.nominal_leg_height,
                    0,
                )
                self.control_foot_position(
                    self.forward_offset[1] - self.hip_forward_offset,
                    self.lateral_offset + self.base_stance_width,
                    self.nominal_leg_height - self.current_foot_lift,
                    1,
                )
            else:
                # right foot = stance
                self.control_foot_position(
                    self.forward_offset[0] - self.hip_forward_offset,
                    -self.lateral_offset - self.base_stance_width,
                    self.nominal_leg_height - self.current_foot_lift,
                    0,
                )
                self.control_foot_position(
                    self.forward_offset[1] - self.hip_forward_offset,
                    self.lateral_offset + self.base_stance_width,
                    self.nominal_leg_height,
                    1,
                )

            if self.step_cycle_counter >= self.step_cycle_length:
                # Reset cycle counter and update offsets
                self.stance_foot_index ^= 1
                self.step_cycle_counter = 1
                self.accumulated_forward_offset = 0.0
                self.previous_stance_foot_offset = self.forward_offset[self.stance_foot_index]
                self.previous_swing_foot_offset = self.forward_offset[self.stance_foot_index ^ 1]
                self.current_foot_lift = 0.0
                self.gait_phase = 20
            else:
                self.step_cycle_counter += 1

    def get_joint_angles(self):
        """
        Return a dictionary with all the joint angles in radians.
        """
        angles = {}
        angles["left_hip_yaw"] = 0.0
        angles["right_hip_yaw"] = 0.0

        angles["left_hip_roll"] = self.K1[0]
        angles["left_hip_pitch"] = -self.K0[0] + -self.hip_pitch_offset
        angles["left_knee"] = self.H[0]
        angles["left_ankle"] = self.A0[0]

        angles["right_hip_roll"] = self.K1[1]
        angles["right_hip_pitch"] = self.K0[1] + self.hip_pitch_offset
        angles["right_knee"] = -self.H[1]
        angles["right_ankle"] = -self.A0[1]

        # Arms & others
        angles["left_shoulder_yaw"] = 0.0
        angles["left_shoulder_pitch"] = 3 * self.K1[0]
        angles["left_elbow"] = 0.0
        angles["left_gripper"] = 0.0

        angles["right_shoulder_yaw"] = 0.0
        angles["right_shoulder_pitch"] = 3 * self.K1[1]  # Use the right hip roll value
        angles["right_elbow"] = self.H[1]
        angles["right_gripper"] = 0.0

        return angles


async def run_walking(host: str = "localhost", port: int = 50051, no_lateral: bool = False) -> None:
    """Run the walking controller in the simulator."""
    logger.info("Starting walking controller...")

    controller = BipedController(lateral_movement_enabled=not no_lateral)
    dt = 0.002  # Reduced from 1000Hz to 500Hz for simulation stability

    async with KOS(ip=host, port=port) as kos:
        # Reset the simulation
        await kos.sim.reset()

        # Configure all actuators with adjusted gains for simulation
        for actuator_id in ACTUATOR_MAPPING.values():
            gains = {
                # Leg joints need higher gains
                31: (100, 10),  # left_hip_yaw
                32: (120, 12),  # left_hip_roll
                33: (150, 15),  # left_hip_pitch
                34: (150, 15),  # left_knee
                35: (100, 10),  # left_ankle
                41: (100, 10),  # right_hip_yaw
                42: (120, 12),  # right_hip_roll
                43: (150, 15),  # right_hip_pitch
                44: (150, 15),  # right_knee
                45: (100, 10),  # right_ankle
                # Arm joints can have lower gains
                11: (50, 5),  # left_shoulder_yaw
                12: (50, 5),  # left_shoulder_pitch
                13: (30, 3),  # left_elbow
                14: (20, 2),  # left_gripper
                21: (50, 5),  # right_shoulder_yaw
                22: (50, 5),  # right_shoulder_pitch
                23: (30, 3),  # right_elbow
                24: (20, 2),  # right_gripper
            }
            kp, kd = gains.get(actuator_id, (32, 32))
            await kos.actuator.configure_actuator(
                actuator_id=actuator_id,
                kp=kp,  # Position gain
                kd=kd,  # Velocity gain
                max_torque=100,  # Increased from 80 for better tracking
                torque_enabled=True,
            )

        # Start from a stable standing position
        initial_pose = [
            {"actuator_id": 33, "position": -15},  # left_hip_pitch
            {"actuator_id": 34, "position": 30},  # left_knee
            {"actuator_id": 35, "position": -15},  # left_ankle
            {"actuator_id": 43, "position": 15},  # right_hip_pitch
            {"actuator_id": 44, "position": -30},  # right_knee
            {"actuator_id": 45, "position": 15},  # right_ankle
        ]
        await kos.actuator.command_actuators(initial_pose)
        await asyncio.sleep(2)  # Give time to reach initial pose

        # Countdown before starting movement
        for i in range(5, 0, -1):
            logger.info(f"Starting in {i}...")
            await asyncio.sleep(1)

        commands_sent = 0
        start_time = time.time()

        try:
            while True:
                # Update the gait state machine
                controller.update_gait()

                # Get joint angles and convert to commands
                angles_dict = controller.get_joint_angles()
                commands = []
                for joint_name, angle_radians in angles_dict.items():
                    if joint_name in ACTUATOR_MAPPING:
                        actuator_id = ACTUATOR_MAPPING[joint_name]
                        angle_degrees = math.degrees(angle_radians)
                        if actuator_id == 32:  # Special case for left_hip_roll
                            angle_degrees = -angle_degrees
                        commands.append({"actuator_id": actuator_id, "position": angle_degrees})

                # Send commands to the robot
                if commands:
                    await kos.actuator.command_actuators(commands)

                # Track commands per second
                commands_sent += 1
                current_time = time.time()
                if current_time - start_time >= 1.0:
                    logger.info(f"Commands per second (CPS): {commands_sent}")
                    commands_sent = 0
                    start_time = current_time

                await asyncio.sleep(dt)

        except KeyboardInterrupt:
            logger.info("Walking controller stopped by user")
            return


async def main() -> None:
    """Main entry point for the walking simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-lateral", action="store_true", help="Disable lateral movements")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)
    await run_walking(host=args.host, port=args.port, no_lateral=args.no_lateral)


if __name__ == "__main__":
    asyncio.run(main())
