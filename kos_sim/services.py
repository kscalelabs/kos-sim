"""Service implementations for MuJoCo simulation."""

import asyncio
import math

import grpc
import numpy as np
from google.protobuf import empty_pb2
from kmv.viewer import MujocoViewerHandler
from kos_protos import (
    actuator_pb2,
    actuator_pb2_grpc,
    common_pb2,
    imu_pb2,
    imu_pb2_grpc,
    process_manager_pb2,
    process_manager_pb2_grpc,
    sim_pb2,
    sim_pb2_grpc,
)

from kos_sim import logger
from kos_sim.simulator import (
    GEOM_TO_MARKER_MAPPING,
    MARKER_TO_GEOM_MAPPING,
    ActuatorCommand,
    ConfigureActuatorRequest,
    MujocoSimulator,
)
from kos_sim.video_recorder import VideoRecorder


class SimService(sim_pb2_grpc.SimulationServiceServicer):
    """Implementation of SimService that wraps a MuJoCo simulation."""

    def __init__(self, simulator: MujocoSimulator) -> None:
        self.simulator = simulator

    async def Reset(  # noqa: N802
        self,
        request: sim_pb2.ResetRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.ActionResponse:
        """Reset the simulation to initial or specified state."""
        try:
            logger.debug("Resetting simulator")
            await self.simulator.reset(
                xyz=None if request.HasField("pos") is None else (request.pos.x, request.pos.y, request.pos.z),
                quat=(
                    None
                    if request.HasField("quat") is None
                    else (request.quat.w, request.quat.x, request.quat.y, request.quat.z)
                ),
                joint_pos=(
                    None
                    if request.joints is None
                    else {joint.name: joint.pos for joint in request.joints.values if joint.HasField("pos")}
                ),
                joint_vel=(
                    None
                    if request.joints is None
                    else {joint.name: joint.vel for joint in request.joints.values if joint.HasField("vel")}
                ),
            )
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
        try:
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
        try:
            params = request.parameters
            if params.HasField("time_scale"):
                logger.debug("Setting time scale to %f", params.time_scale)
                self.simulator._model.opt.timestep = self.simulator._dt / params.time_scale
            if params.HasField("gravity"):
                logger.debug("Setting gravity to %f", params.gravity)
                self.simulator._model.opt.gravity[2] = params.gravity
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

    async def GetMarkers(  # noqa: N802
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> sim_pb2.GetMarkersResponse:
        """Get all markers in the simulation."""
        markers = []
        for marker in self.simulator._markers.values():
            assert marker.geom is not None
            assert marker.tracking_cfg is not None

            geom_name = GEOM_TO_MARKER_MAPPING[marker.geom]

            markers.append(
                sim_pb2.Marker(
                    name=marker.name,
                    marker_type=getattr(sim_pb2.Marker.MarkerType, geom_name),
                    target_name=marker.tracking_cfg.target_name,
                    target_type=getattr(sim_pb2.Marker.TargetType, str.upper(marker.tracking_cfg.target_type)),
                    scale=sim_pb2.Marker.Scale(scale=marker.scale),
                    color=sim_pb2.Marker.RGBA(
                        r=marker.color[0], g=marker.color[1], b=marker.color[2], a=marker.color[3]
                    ),
                    label=(marker.label is not None),
                    track_rotation=marker.tracking_cfg.track_rotation,
                    offset=sim_pb2.Marker.Offset(
                        x=marker.tracking_cfg.offset[0],
                        y=marker.tracking_cfg.offset[1],
                        z=marker.tracking_cfg.offset[2],
                    ),
                )
            )
        return sim_pb2.GetMarkersResponse(markers=markers)

    async def AddMarker(  # noqa: N802
        self, marker: sim_pb2.Marker, context: grpc.ServicerContext
    ) -> common_pb2.ActionResponse:
        """Add a marker to the simulation."""
        if marker.name in self.simulator._markers:
            logger.warning("Marker %s already exists. Not adding again.", marker.name)
            return common_pb2.ActionResponse(success=False, error="Marker already exists")

        target_type = sim_pb2.Marker.TargetType.Name(marker.target_type)

        marker_params = {
            "name": marker.name,
            "model": self.simulator._model,
            "geom": MARKER_TO_GEOM_MAPPING[sim_pb2.Marker.MarkerType.Name(marker.marker_type)],
            "scale": np.array(marker.scale.scale),
            "color": np.array([marker.color.r, marker.color.g, marker.color.b, marker.color.a]),
            "tracking_offset": np.array([marker.offset.x, marker.offset.y, marker.offset.z]),
            "track_rotation": marker.track_rotation,
        }

        if target_type == "BODY":
            marker_params["track_body_name"] = marker.target_name
        elif target_type == "SITE":
            marker_params["track_site_name"] = marker.target_name

        if marker.label:
            marker_params["label"] = marker.name

        self.simulator._markers[marker.name] = MujocoViewerHandler.create_marker(**marker_params)
        return common_pb2.ActionResponse(success=True)

    async def RemoveMarker(  # noqa: N802
        self, request: sim_pb2.RemoveMarkerRequest, context: grpc.ServicerContext
    ) -> common_pb2.ActionResponse:
        """Remove a marker from the simulation."""
        if request.name not in self.simulator._markers:
            logger.warning("Marker %s not found. Not removing.", request.name)
            return common_pb2.ActionResponse(
                success=False, error=common_pb2.Error(code=common_pb2.ErrorCode.NOT_FOUND, message="Marker not found")
            )

        del self.simulator._markers[request.name]
        return common_pb2.ActionResponse(success=True)

    async def UpdateMarker(  # noqa: N802
        self, request: sim_pb2.UpdateMarkerRequest, context: grpc.ServicerContext
    ) -> common_pb2.ActionResponse:
        """Update a marker in the simulation."""
        if request.name not in self.simulator._markers:
            logger.warning("Marker %s not found. Not updating.", request.name)
            return common_pb2.ActionResponse(
                success=False, error=common_pb2.Error(code=common_pb2.ErrorCode.NOT_FOUND, message="Marker not found")
            )

        existing_marker = self.simulator._markers[request.name]

        assert existing_marker.tracking_cfg is not None

        if request.HasField("marker_type"):
            existing_marker.geom = MARKER_TO_GEOM_MAPPING[sim_pb2.Marker.MarkerType.Name(request.marker_type)]
        if request.HasField("offset"):
            existing_marker.tracking_cfg.offset = np.array([request.offset.x, request.offset.y, request.offset.z])
        if request.HasField("color"):
            existing_marker.color = np.array([request.color.r, request.color.g, request.color.b, request.color.a])
        if request.HasField("scale"):
            existing_marker.scale = np.array(request.scale.scale)
        if request.HasField("label"):
            existing_marker.label = request.name if request.label else None

        return common_pb2.ActionResponse(success=True)


class ActuatorService(actuator_pb2_grpc.ActuatorServiceServicer):
    """Implementation of ActuatorService that wraps a MuJoCo simulation."""

    def __init__(self, simulator: MujocoSimulator, step_lock: asyncio.Semaphore) -> None:
        self.simulator = simulator
        self.step_lock = step_lock

    async def CommandActuators(  # noqa: N802
        self,
        request: actuator_pb2.CommandActuatorsRequest,
        context: grpc.ServicerContext,
    ) -> actuator_pb2.CommandActuatorsResponse:
        """Implements CommandActuators by forwarding to simulator."""
        try:
            # Convert degrees to radians.
            commands: dict[int, ActuatorCommand] = {
                cmd.actuator_id: {
                    "position": math.radians(cmd.position),
                    "velocity": math.radians(cmd.velocity),
                    "torque": cmd.torque,
                }
                for cmd in request.commands
            }
            async with self.step_lock:
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
        ids = request.actuator_ids or list(self.simulator._joint_id_to_name.keys())
        try:
            states = {joint_id: await self.simulator.get_actuator_state(joint_id) for joint_id in ids}
            return actuator_pb2.GetActuatorsStateResponse(
                states=[
                    actuator_pb2.ActuatorStateResponse(
                        actuator_id=joint_id,
                        position=math.degrees(state.position),
                        velocity=math.degrees(state.velocity),
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
        acc_name: str | None = "imu_acc",
        gyro_name: str | None = "imu_gyro",
        mag_name: str | None = "imu_mag",
        quat_name: str | None = "base_link_quat",
    ) -> None:
        self.simulator = simulator
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


class ProcessManagerService(process_manager_pb2_grpc.ProcessManagerServiceServicer):
    """Implementation of ProcessManagerService that wraps a MuJoCo simulation."""

    def __init__(self, simulator: MujocoSimulator, video_recorder: VideoRecorder) -> None:
        self.simulator = simulator
        self.video_recorder = video_recorder

    async def StartKClip(  # noqa: N802
        self, request: process_manager_pb2.KClipStartRequest, context: grpc.ServicerContext
    ) -> process_manager_pb2.KClipStartResponse:
        """Implements StartKClip by starting k-clip recording."""
        if self.video_recorder is None:
            return process_manager_pb2.KClipStartResponse(
                error=common_pb2.Error(
                    code=common_pb2.ErrorCode.INVALID_ARGUMENT,
                    message="`video_recorder` is `None`. Video recording not enabled.",
                )
            )
        clip_id = self.video_recorder.start_recording()
        return process_manager_pb2.KClipStartResponse(clip_uuid=clip_id)

    async def StopKClip(  # noqa: N802
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> process_manager_pb2.KClipStopResponse:
        """Implements StopKClip by stopping k-clip recording."""
        if self.video_recorder is None:
            return process_manager_pb2.KClipStopResponse(
                error=common_pb2.Error(
                    code=common_pb2.ErrorCode.INVALID_ARGUMENT,
                    message="`video_recorder` is `None`. Video recording not enabled.",
                )
            )
        self.video_recorder.stop_recording()
        return process_manager_pb2.KClipStopResponse()
