"""Service implementations for MuJoCo simulation."""

import math

import grpc
from google.protobuf import empty_pb2
from kos_protos import (
    actuator_pb2,
    actuator_pb2_grpc,
    common_pb2,
    imu_pb2,
    imu_pb2_grpc,
    sim_pb2,
    sim_pb2_grpc,
)

from kos_sim import logger
from kos_sim.simulator import ConfigureActuatorRequest, MujocoSimulator
from kos_sim.stepping import StepController


class SimService(sim_pb2_grpc.SimulationServiceServicer):
    """Implementation of SimService that wraps a MuJoCo simulation."""

    def __init__(self, simulator: MujocoSimulator, step_controller: StepController) -> None:
        self.simulator = simulator
        self.step_controller = step_controller

    async def Reset(  # noqa: N802
        self,
        request: sim_pb2.ResetRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.ActionResponse:
        """Reset the simulation to initial or specified state."""
        logger.info("Reset request received")
        await self.step_controller.set_paused(True)
        try:
            if request.HasField("initial_state"):
                qpos = list(request.initial_state.qpos)
                logger.debug("Resetting with qpos: %s", qpos)
                await self.simulator.reset(qpos=qpos)
            else:
                logger.debug("Resetting to default state")
                await self.simulator.reset()
            await self.step_controller.set_paused(False)
            return common_pb2.ActionResponse(success=True)
        except Exception as e:
            logger.error("Reset failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return common_pb2.ActionResponse(success=False, error=str(e))

    async def SetPaused(  # noqa: N802
        self,
        request: sim_pb2.SetPausedRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.ActionResponse:  # noqa: N802
        """Pause or unpause the simulation."""
        logger.info("SetPaused request received: paused=%s", request.paused)
        try:
            await self.step_controller.set_paused(request.paused)
            return common_pb2.ActionResponse(success=True)
        except Exception as e:
            logger.error("SetPaused failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return common_pb2.ActionResponse(success=False, error=str(e))

    async def Step(  # noqa: N802
        self,
        request: sim_pb2.StepRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.ActionResponse:
        raise NotImplementedError("Step is not implemented")

    async def SetParameters(  # noqa: N802
        self,
        request: sim_pb2.SetParametersRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.ActionResponse:
        """Set simulation parameters."""
        logger.info("SetParameters request received: %s", request)
        try:
            await self.step_controller.set_paused(True)
            params = request.parameters
            if params.HasField("time_scale"):
                logger.debug("Setting time scale to %f", params.time_scale)
                self.simulator._model.opt.timestep = self.simulator._dt / params.time_scale
            if params.HasField("gravity"):
                logger.debug("Setting gravity to %f", params.gravity)
                self.simulator._model.opt.gravity[2] = params.gravity
            if params.HasField("initial_state"):
                logger.debug("Setting initial state: %s", params.initial_state)
                qpos = list(params.initial_state.qpos)
                await self.simulator.reset(qpos)
            await self.step_controller.set_paused(False)
            return common_pb2.ActionResponse(success=True)
        except Exception as e:
            logger.error("SetParameters failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return common_pb2.ActionResponse(success=False, error=str(e))

    async def GetParameters(  # noqa: N802
        self,
        request: empty_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> sim_pb2.GetParametersResponse:
        """Get current simulation parameters."""
        logger.info("GetParameters request received")
        try:
            params = sim_pb2.SimulationParameters(
                time_scale=self.simulator._dt / self.simulator._model.opt.timestep,
                gravity=float(self.simulator._model.opt.gravity[2]),
            )
            logger.debug("Current parameters: time_scale=%f, gravity=%f", params.time_scale, params.gravity)
            return sim_pb2.GetParametersResponse(parameters=params)
        except Exception as e:
            logger.error("GetParameters failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sim_pb2.GetParametersResponse(error=common_pb2.Error(message=str(e)))


class ActuatorService(actuator_pb2_grpc.ActuatorServiceServicer):
    """Implementation of ActuatorService that wraps a MuJoCo simulation."""

    def __init__(self, simulator: MujocoSimulator, step_controller: StepController) -> None:
        self.simulator = simulator
        self.step_controller = step_controller

    async def CommandActuators(  # noqa: N802
        self,
        request: actuator_pb2.CommandActuatorsRequest,
        context: grpc.ServicerContext,
    ) -> actuator_pb2.CommandActuatorsResponse:
        """Implements CommandActuators by forwarding to simulator."""
        try:
            # Convert degrees to radians.
            commands = {cmd.actuator_id: math.radians(cmd.position) for cmd in request.commands}
            await self.simulator.command_actuators(commands)
            return actuator_pb2.CommandActuatorsResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return actuator_pb2.CommandActuatorsResponse()

    async def GetActuatorsState(  # noqa: N802
        self,
        request: actuator_pb2.GetActuatorsStateRequest,
        context: grpc.ServicerContext,
    ) -> actuator_pb2.GetActuatorsStateResponse:
        """Implements GetActuatorsState by reading from simulator."""
        logger.info("GetActuatorsState request received")
        ids = request.actuator_ids or list(self.simulator._joint_id_to_name.keys())
        try:
            states = {joint_id: await self.simulator.get_actuator_state(joint_id) for joint_id in ids}
            return actuator_pb2.GetActuatorsStateResponse(
                states=[
                    actuator_pb2.ActuatorStateResponse(
                        actuator_id=joint_id,
                        position=math.degrees(state.position),
                        velocity=math.degrees(state.velocity),
                        torque=state.effort,
                        online=True,
                    )
                    for joint_id, state in states.items()
                ]
            )
        except Exception as e:
            logger.error("GetActuatorsState failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return actuator_pb2.GetActuatorsStateResponse()

    async def ConfigureActuator(  # noqa: N802
        self,
        request: actuator_pb2.ConfigureActuatorRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.ActionResponse:
        configuration: ConfigureActuatorRequest = {}
        if request.HasField("torque_enabled"):
            configuration["torque_enabled"] = request.torque_enabled
        if request.HasField("zero_position"):
            configuration["zero_position"] = request.zero_position
        if request.HasField("kp"):
            configuration["kp"] = request.kp
        if request.HasField("kd"):
            configuration["kd"] = request.kd
        if request.HasField("max_torque"):
            configuration["max_torque"] = request.max_torque
        await self.simulator.configure_actuator(request.actuator_id, configuration)

        return common_pb2.ActionResponse(success=True)


class IMUService(imu_pb2_grpc.IMUServiceServicer):
    """Implementation of IMUService that wraps a MuJoCo simulation."""

    def __init__(
        self,
        simulator: MujocoSimulator,
        step_controller: StepController,
        acc_name: str | None = "imu_acc",
        gyro_name: str | None = "imu_gyro",
        mag_name: str | None = "imu_mag",
        quat_name: str | None = "base_link_quat",
    ) -> None:
        self.simulator = simulator
        self.step_controller = step_controller
        self.acc_name = acc_name
        self.gyro_name = gyro_name
        self.mag_name = mag_name
        self.quat_name = quat_name

    async def GetValues(  # noqa: N802
        self,
        request: empty_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> imu_pb2.IMUValuesResponse:
        """Implements GetValues by reading IMU sensor data from simulator."""
        try:
            if self.acc_name is None or self.gyro_name is None:
                raise ValueError("Accelerometer or gyroscope name not set")
            acc_data = await self.simulator.get_sensor_data(self.acc_name)
            gyro_data = await self.simulator.get_sensor_data(self.gyro_name)
            mag_data = None if self.mag_name is None else await self.simulator.get_sensor_data(self.mag_name)
            return imu_pb2.IMUValuesResponse(
                accel_x=float(acc_data[0]),
                accel_y=float(acc_data[1]),
                accel_z=float(acc_data[2]),
                gyro_x=float(gyro_data[0]),
                gyro_y=float(gyro_data[1]),
                gyro_z=float(gyro_data[2]),
                mag_x=None if mag_data is None else float(mag_data[0]),
                mag_y=None if mag_data is None else float(mag_data[1]),
                mag_z=None if mag_data is None else float(mag_data[2]),
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return imu_pb2.IMUValuesResponse()

    async def GetQuaternion(  # noqa: N802
        self,
        request: empty_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> imu_pb2.QuaternionResponse:
        """Implements GetQuaternion by reading orientation data from simulator."""
        try:
            if self.quat_name is None:
                raise ValueError("Quaternion name not set")
            quat_data = await self.simulator.get_sensor_data(self.quat_name)
            return imu_pb2.QuaternionResponse(
                w=float(quat_data[0]), x=float(quat_data[1]), y=float(quat_data[2]), z=float(quat_data[3])
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return imu_pb2.QuaternionResponse()

    async def GetEuler(  # noqa: N802
        self,
        request: empty_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> imu_pb2.EulerAnglesResponse:
        """Implements GetEuler by converting orientation quaternion to Euler angles."""
        logger.info("GetEuler request received")
        try:
            if self.quat_name is None:
                raise ValueError("Quaternion name not set")
            quat_data = await self.simulator.get_sensor_data(self.quat_name)
            # Extract quaternion components
            w, x, y, z = [float(q) for q in quat_data]

            # Convert quaternion to Euler angles (roll, pitch, yaw)
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            # Pitch (y-axis rotation)
            sinp = 2 * (w * y - z * x)
            pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)

            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            # Convert to degrees
            roll_deg = math.degrees(roll)
            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)

            return imu_pb2.EulerAnglesResponse(roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return imu_pb2.EulerAnglesResponse()
