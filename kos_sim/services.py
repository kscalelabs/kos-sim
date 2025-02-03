"""Service implementations for MuJoCo simulation."""

import math

import grpc
from google.protobuf import empty_pb2
from kos_protos import actuator_pb2, actuator_pb2_grpc, common_pb2, imu_pb2, imu_pb2_grpc, sim_pb2, sim_pb2_grpc

from kos_sim import logger
from kos_sim.mujoco_simulator import MujocoSimulator
from kos_sim.stepping import StepController


class SimService(sim_pb2_grpc.SimulationServiceServicer):
    """Implementation of SimService that wraps a MuJoCo simulation."""

    def __init__(self, simulator: MujocoSimulator, step_controller: StepController) -> None:
        self.simulator = simulator
        self.step_controller = step_controller

    def Reset(  # noqa: N802
        self, request: sim_pb2.ResetRequest, context: grpc.ServicerContext
    ) -> common_pb2.ActionResponse:  # noqa: N802
        """Reset the simulation to initial or specified state."""
        logger.info("Reset request received: %s", request)
        self.step_controller.set_paused(True)
        try:
            if request.HasField("initial_state"):
                qpos = list(request.initial_state.qpos)
                logger.debug("Resetting with qpos: %s", qpos)
                self.simulator.reset(qpos=qpos)
            else:
                logger.debug("Resetting to default state")
                self.simulator.reset()
            self.step_controller.set_paused(False)
            return common_pb2.ActionResponse(success=True)
        except Exception as e:
            logger.error("Reset failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return common_pb2.ActionResponse(success=False, error=str(e))

    def SetPaused(  # noqa: N802
        self, request: sim_pb2.SetPausedRequest, context: grpc.ServicerContext
    ) -> common_pb2.ActionResponse:  # noqa: N802
        """Pause or unpause the simulation."""
        logger.info("SetPaused request received: paused=%s", request.paused)
        try:
            self.step_controller.set_paused(request.paused)
            return common_pb2.ActionResponse(success=True)
        except Exception as e:
            logger.error("SetPaused failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return common_pb2.ActionResponse(success=False, error=str(e))

    def Step(  # noqa: N802
        self, request: sim_pb2.StepRequest, context: grpc.ServicerContext
    ) -> common_pb2.ActionResponse:  # noqa: N802
        """Step the simulation forward."""
        logger.info(
            "Step request received: num_steps=%d, step_size=%s",
            request.num_steps,
            request.step_size if request.HasField("step_size") else "default",
        )
        try:
            if request.HasField("step_size"):
                original_dt = self.simulator._model.opt.timestep
                self.simulator._model.opt.timestep = request.step_size

            self.step_controller.request_steps(request.num_steps)

            if request.HasField("step_size"):
                self.simulator._model.opt.timestep = original_dt

            return common_pb2.ActionResponse(success=True)
        except Exception as e:
            logger.error("Step failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return common_pb2.ActionResponse(success=False, error=str(e))

    def SetParameters(  # noqa: N802
        self, request: sim_pb2.SetParametersRequest, context: grpc.ServicerContext
    ) -> common_pb2.ActionResponse:
        """Set simulation parameters."""
        logger.info("SetParameters request received: %s", request)
        try:
            self.step_controller.set_paused(True)
            params = request.parameters
            if params.HasField("time_scale"):
                logger.debug("Setting time scale to %f", params.time_scale)
                self.simulator._model.opt.timestep = self.simulator._config.dt / params.time_scale
            if params.HasField("gravity"):
                logger.debug("Setting gravity to %f", params.gravity)
                self.simulator._model.opt.gravity[2] = params.gravity
            if params.HasField("initial_state"):
                logger.debug("Setting initial state: %s", params.initial_state)
                qpos = list(params.initial_state.qpos)
                self.simulator.reset(position=None, orientation=qpos[3:7] if len(qpos) >= 7 else None)
            self.step_controller.set_paused(False)
            return common_pb2.ActionResponse(success=True)
        except Exception as e:
            logger.error("SetParameters failed: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return common_pb2.ActionResponse(success=False, error=str(e))

    def GetParameters(  # noqa: N802
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> sim_pb2.GetParametersResponse:
        """Get current simulation parameters."""
        logger.info("GetParameters request received")
        try:
            params = sim_pb2.SimulationParameters(
                time_scale=self.simulator._config.dt / self.simulator._model.opt.timestep,
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

    def CommandActuators(  # noqa: N802
        self, request: actuator_pb2.CommandActuatorsRequest, context: grpc.ServicerContext
    ) -> actuator_pb2.CommandActuatorsResponse:
        """Implements CommandActuators by forwarding to simulator."""
        try:
            commands = {cmd.actuator_id: math.radians(cmd.position) for cmd in request.commands}
            self.simulator.command_actuators(commands)
            return actuator_pb2.CommandActuatorsResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return actuator_pb2.CommandActuatorsResponse()

    def GetActuatorsState(  # noqa: N802
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> actuator_pb2.GetActuatorsStateResponse:
        """Implements GetActuatorsState by reading from simulator."""
        try:
            states = {
                joint_id: self.simulator.get_actuator_state(joint_id)
                for joint_id in self.simulator._config.joint_id_to_name.keys()
            }
            return actuator_pb2.GetActuatorsStateResponse(
                states=[
                    actuator_pb2.ActuatorStateResponse(
                        actuator_id=joint_id, position=math.degrees(float(state)), online=True
                    )
                    for joint_id, state in states.items()
                ]
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return actuator_pb2.GetActuatorsStateResponse()
        
    def ConfigureActuator(self, request: actuator_pb2.ConfigureActuatorRequest, context: grpc.ServicerContext) -> common_pb2.ActionResponse:
        configuration = {}
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
        self.simulator.configure_actuator(request.actuator_id, configuration)

        return common_pb2.ActionResponse(success=True)


class IMUService(imu_pb2_grpc.IMUServiceServicer):
    """Implementation of IMUService that wraps a MuJoCo simulation."""

    def __init__(self, simulator: MujocoSimulator, step_controller: StepController) -> None:
        self.simulator = simulator
        self.step_controller = step_controller

    def GetValues(  # noqa: N802
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> imu_pb2.IMUValuesResponse:
        """Implements GetValues by reading IMU sensor data from simulator."""
        try:
            imu_data = self.simulator.get_sensor_data("imu")
            return imu_pb2.IMUValuesResponse(
                accelerometer=imu_pb2.Vector3(x=float(imu_data[0]), y=float(imu_data[1]), z=float(imu_data[2])),
                gyroscope=imu_pb2.Vector3(x=float(imu_data[3]), y=float(imu_data[4]), z=float(imu_data[5])),
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return imu_pb2.IMUValuesResponse()

    def GetQuaternion(  # noqa: N802
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> imu_pb2.QuaternionResponse:
        """Implements GetQuaternion by reading orientation data from simulator."""
        try:
            quat_data = self.simulator.get_sensor_data("base_link_quat")
            return imu_pb2.QuaternionResponse(
                w=float(quat_data[0]), x=float(quat_data[1]), y=float(quat_data[2]), z=float(quat_data[3])
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return imu_pb2.QuaternionResponse()

    def GetEuler(  # noqa: N802
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> imu_pb2.EulerAnglesResponse:
        """Implements GetEuler by converting orientation quaternion to Euler angles."""
        try:
            quat_data = self.simulator.get_sensor_data("base_link_quat")
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
            
            return imu_pb2.EulerAnglesResponse(
                roll=roll_deg,
                pitch=pitch_deg,
                yaw=yaw_deg
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return imu_pb2.EulerAnglesResponse()