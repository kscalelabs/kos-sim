"""Service implementations for MuJoCo simulation."""

import math

import grpc
from google.protobuf import empty_pb2
from kos_protos import actuator_pb2, actuator_pb2_grpc, imu_pb2, imu_pb2_grpc

from kos_sim.simulator import MujocoSimulator


class ActuatorService(actuator_pb2_grpc.ActuatorServiceServicer):
    """Implementation of ActuatorService that wraps a MuJoCo simulation."""

    def __init__(self, simulator: MujocoSimulator) -> None:
        self.simulator = simulator

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
                        actuator_id=joint_id,
                        position=math.degrees(float(state)),
                        online=True
                    )
                    for joint_id, state in states.items()
                ]
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return actuator_pb2.GetActuatorsStateResponse()


class IMUService(imu_pb2_grpc.IMUServiceServicer):
    """Implementation of IMUService that wraps a MuJoCo simulation."""

    def __init__(self, simulator: MujocoSimulator) -> None:
        self.simulator = simulator

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
            quat_data = self.simulator.get_sensor_data("orientation")
            return imu_pb2.QuaternionResponse(
                w=float(quat_data[0]), x=float(quat_data[1]), y=float(quat_data[2]), z=float(quat_data[3])
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return imu_pb2.QuaternionResponse()
