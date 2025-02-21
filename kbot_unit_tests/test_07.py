"""This script is used to test the robot's ability to walk in a straight line."""

import argparse
import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import colorlogging
import numpy as np
import onnx
import onnxruntime as ort
import yaml
from pykos import KOS
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


async def load_policy_and_config(checkpoint_path: str) -> tuple[ort.InferenceSession, dict]:
    try:
        # List YAML and ONNX files in the given directory
        yaml_files = [f for f in os.listdir(checkpoint_path) if f.endswith("env.yaml")]
        policy_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".onnx")]

        if not yaml_files or not policy_files:
            raise FileNotFoundError(f"Could not find env.yaml and .onnx files in {checkpoint_path}")

        yaml_file = yaml_files[0]  # Use the first found YAML file
        policy_file = policy_files[0]  # Use the first found ONNX file

        policy_path = os.path.join(checkpoint_path, policy_file)
        yaml_path = os.path.join(checkpoint_path, yaml_file)

        logger.info("Loading policy from: %s", os.path.abspath(policy_path))
        logger.info("Loading config from: %s", os.path.abspath(yaml_path))

        # Load the policy and create an inference session
        policy = onnx.load(policy_path)
        session = ort.InferenceSession(policy.SerializeToString())

        with open(yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

        logger.info("Loaded policy and config")

        return session, config

    except Exception as e:
        logger.error("Error in load_policy_and_config: %s", e)
        raise


isaac_joint_names = [
    "left_hip_pitch_04",
    "left_shoulder_pitch_03",
    "right_hip_pitch_04",
    "right_shoulder_pitch_03",
    "left_hip_roll_03",
    "left_shoulder_roll_03",
    "right_hip_roll_03",
    "right_shoulder_roll_03",
    "left_hip_yaw_03",
    "left_shoulder_yaw_02",
    "right_hip_yaw_03",
    "right_shoulder_yaw_02",
    "left_knee_04",
    "left_elbow_02",
    "right_knee_04",
    "right_elbow_02",
    "left_ankle_02",
    "left_wrist_02",
    "right_ankle_02",
    "right_wrist_02",
]

start_pos = {
    "left_shoulder_pitch_03": 0.0,
    "left_shoulder_roll_03": 0.0,
    "left_shoulder_yaw_02": 0.0,
    "left_elbow_02": -math.radians(90),
    "left_wrist_02": 0.0,
    "right_shoulder_pitch_03": 0.0,
    "right_shoulder_roll_03": 0.0,
    "right_shoulder_yaw_02": 0.0,
    "right_elbow_02": math.radians(90),
    "right_wrist_02": 0.0,
    "left_hip_pitch_04": math.radians(20),  # Slight forward lean
    "left_hip_roll_03": 0.0,  # Neutral stance
    "left_hip_yaw_03": 0.0,
    "left_knee_04": math.radians(40),  # Bent knees for stability
    "left_ankle_02": -math.radians(20),  # Compensate for knee bend
    "right_hip_pitch_04": -math.radians(20),
    "right_hip_roll_03": 0.0,
    "right_hip_yaw_03": 0.0,
    "right_knee_04": -math.radians(40),
    "right_ankle_02": math.radians(20),
}


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


ACTUATOR_LIST: list[Actuator] = [
    Actuator(11, 1, 85.0, 2.0, 60.0, "left_shoulder_pitch_03"),
    Actuator(12, 5, 85.0, 2.0, 60.0, "left_shoulder_roll_03"), 
    Actuator(13, 9, 80.0, 1.0, 17.0, "left_shoulder_yaw_02"),
    Actuator(14, 13, 80.0, 1.0, 17.0, "left_elbow_02"),
    Actuator(15, 17, 80.0, 1.0, 17.0, "left_wrist_02"),
    Actuator(21, 3, 85.0, 2.0, 60.0, "right_shoulder_pitch_03"),
    Actuator(22, 7, 85.0, 2.0, 60.0, "right_shoulder_roll_03"),
    Actuator(23, 11, 80.0, 1.0, 17.0, "right_shoulder_yaw_02"), 
    Actuator(24, 15, 80.0, 1.0, 17.0, "right_elbow_02"),
    Actuator(25, 19, 80.0, 1.0, 17.0, "right_wrist_02"),
    Actuator(31, 0, 100.0, 7.0, 80.0, "left_hip_pitch_04"),
    Actuator(32, 4, 85.0, 2.0, 60.0, "left_hip_roll_03"),
    Actuator(33, 8, 85.0, 2.0, 60.0, "left_hip_yaw_03"),
    Actuator(34, 12, 100.0, 7.0, 80.0, "left_knee_04"),
    Actuator(35, 16, 80.0, 1.0, 17.0, "left_ankle_02"),
    Actuator(41, 2, 100.0, 7.0, 80.0, "right_hip_pitch_04"),
    Actuator(42, 6, 85.0, 2.0, 60.0, "right_hip_roll_03"),
    Actuator(43, 10, 85.0, 2.0, 60.0, "right_hip_yaw_03"),
    Actuator(44, 14, 100.0, 7.0, 80.0, "right_knee_04"),
    Actuator(45, 18, 80.0, 1.0, 17.0, "right_ankle_02"),
]


async def configure_robot(kos: KOS) -> None:
    for actuator in ACTUATOR_LIST:
        await kos.actuator.configure_actuator(
            actuator_id=actuator.actuator_id,
            kp=actuator.kp,
            kd=actuator.kd,
            max_torque=actuator.max_torque,
            torque_enabled=True,
        )


class RobotState:
    """Tracks robot state and handles offsets."""

    def __init__(
        self,
        joint_names: list[str],
        start_pos: dict[str, float],
        vel_cmd: np.ndarray,
        config: dict,
        policy: ort.InferenceSession,
        idtoname: dict[int, str],
        nametoid: dict[str, int],
        indextoname: dict[int, str],
        nametoindex: dict[str, int],
    ):
        self.joint_offsets = {name: 0.0 for name in joint_names}
        self.orn_offset = None
        self.start_pos = start_pos
        self.vel_cmd = vel_cmd
        self.config = config
        self.policy = policy
        self.idtoname = idtoname
        self.nametoid = nametoid
        self.indextoname = indextoname
        self.nametoindex = nametoindex
        self.prev_action = np.zeros(len(joint_names), dtype=np.float32)

    async def offset_in_place(self, kos: KOS, joint_names: list[str] | None = None) -> None:
        """Capture current position as zero offset."""
        # Get current joint positions (in degrees)
        # states = await kos.actuator.get_actuators_state([JOINT_NAME_TO_ID[name] for name in joint_names])
        # current_positions = {name: states.states[i].position for i, name in enumerate(joint_names)}

        # # Store negative of current positions as offsets (in degrees)
        # self.joint_offsets = {name: 0.0 for name, _ in current_positions.items()}

        # # Store IMU offset
        # imu_data = await kos.imu.get_euler_angles()
        # initial_quat = R.from_euler("xyz", [imu_data.roll, imu_data.pitch, imu_data.yaw], degrees=True).as_quat()
        # self.orn_offset = R.from_quat(initial_quat).inv()

        pass

    def get_gravity_vector(self, euler_data: Any, imu_sensor_data: Any) -> np.ndarray:
        """Get gravity vector from IMU data."""
        # Process IMU data with offset compensation
        current_quat = R.from_euler("xyz", [euler_data.roll, euler_data.pitch, euler_data.yaw], degrees=True).as_quat()
        if self.orn_offset is not None:
            current_rot = R.from_quat(current_quat)
            quat = (self.orn_offset * current_rot).as_quat()
        else:
            quat = current_quat

        # Calculate gravity vector with offset compensation
        r = R.from_quat(quat)
        gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True)

        # Get angular velocity
        gyro_x = imu_sensor_data.gyro_x or 0.0
        gyro_y = imu_sensor_data.gyro_y or 0.0
        gyro_z = imu_sensor_data.gyro_z or 0.0
        # omega = np.deg2rad(np.array([-gyro_z, -gyro_y, gyro_x]))
        omega = np.deg2rad(np.array([gyro_x, gyro_y, gyro_z]))
        return gvec, quat, omega

    def map_kos_sim_to_isaac(self, states: list[Any], pos: bool = True) -> np.ndarray:
        """Map KOS state to Isaac state."""
        result = np.zeros(len(isaac_joint_names))

        for i in range(len(isaac_joint_names)):
            curr_id = states.states[i].actuator_id
            curr_name = self.idtoname[curr_id]
            curr_value = states.states[i].position if pos else states.states[i].velocity
            curr_value_rad = np.deg2rad(curr_value)

            if pos:
                result[self.nametoindex[curr_name]] = curr_value_rad - self.start_pos[curr_name]
            else:
                result[self.nametoindex[curr_name]] = curr_value_rad

            logger.debug(
                "{'pos' if pos else 'vel'} obs i: %s, id: %s, name: %s,  value: %s",
                i,
                curr_id,
                curr_name,
                curr_value,
            )
        return result

    def map_isaac_to_kos_sim(self, actions: np.ndarray, q: np.ndarray) -> list[dict]:
        """Map Isaac state to KOS state."""
        commands = []

        for i in range(len(actions)):
            curr_name = self.indextoname[i]
            curr_id = self.nametoid[curr_name]
            if "relative_joint_pos" in self.config["actions"]:
                curr_pos = q[self.nametoindex[curr_name]]
                target_pos = np.rad2deg(actions[i] + curr_pos)
            elif "joint_pos" in self.config["actions"]:
                target_pos = np.rad2deg(actions[i])
            else:
                raise ValueError("No action scaling found in config")
            commands.append({"actuator_id": curr_id, "position": target_pos})
        return commands
    
    async def gather_kos_data(self, kos: KOS) -> tuple[list[Any], Any, Any]:
        """Gather data from KOS."""
        return await asyncio.gather(
            kos.actuator.get_actuators_state([actuator.actuator_id for actuator in ACTUATOR_LIST]),
            kos.imu.get_euler_angles(),
            kos.imu.get_imu_values(),
        )

    def get_obs(
        self,
        states: list[Any],
        euler_data: Any,
        imu_sensor_data: Any,
        prev_action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get robot state with offset compensation."""
        # Batch state requests

        q = self.map_kos_sim_to_isaac(states, pos=True)
        dq = self.map_kos_sim_to_isaac(states, pos=False)

        gvec, quat, omega = self.get_gravity_vector(euler_data, imu_sensor_data)

        logger.debug("vel_cmd: %s", self.vel_cmd)
        logger.debug("gvec: %s", gvec)
        logger.debug("q: %s", q)
        logger.debug("dq: %s", dq)
        logger.debug("prev_action: %s", prev_action)



        return self.vel_cmd, gvec, q, dq, prev_action


    async def inner_loop(self, kos: KOS) -> None:
        states, euler_data, imu_sensor_data = await self.gather_kos_data(kos)
        vel_cmd, gvec, q, dq, prev_action = self.get_obs(states, euler_data, imu_sensor_data, self.prev_action)
        obs = np.concatenate([vel_cmd, gvec, q, dq, prev_action])
        input_name = self.policy.get_inputs()[0].name
        actions_raw = self.policy.run(None, {input_name: obs.reshape(1, -1).astype(np.float32)})[0][0]  # shape (20,)
        self.prev_action = actions_raw

        # check if joint_pos or relative_joint_pos is in the config
        if "joint_pos" in self.config["actions"]:
            actions = actions_raw * self.config["actions"]["joint_pos"]["scale"]
        elif "relative_joint_pos" in self.config["actions"]:
            actions = actions_raw * self.config["actions"]["relative_joint_pos"]["scale"]
        else:
            raise ValueError("No action scaling found in config")
        
        logger.debug("actions: %s", actions)

        commands = self.map_isaac_to_kos_sim(actions, q)
        for cmd in commands:
            logger.debug("  %s: %s deg", cmd["actuator_id"], cmd["position"])

        await kos.actuator.command_actuators(commands)


async def run_robot(args: argparse.Namespace) -> None:
    session, config = await load_policy_and_config(args.checkpoint_path)

    # Create mappings between actuator ID and joint name
    idtoname = {actuator.actuator_id: actuator.joint_name for actuator in ACTUATOR_LIST}
    nametoid = {actuator.joint_name: actuator.actuator_id for actuator in ACTUATOR_LIST}

    # Example policy joint mapping (assumes policy outputs match the order of joints)
    # Assuming your policy output corresponds to the joints in a specific order (as per ACTUATOR_LIST)
    indextoname = {i: isaac_joint_names[i] for i in range(len(isaac_joint_names))}
    nametoindex = {isaac_joint_names[i]: i for i in range(len(isaac_joint_names))}

    vel_cmd = np.array(args.vel_cmd.split(","), dtype=np.float32)
    robot_state = RobotState(
        joint_names=isaac_joint_names,
        start_pos=start_pos,
        vel_cmd=vel_cmd,
        config=config,
        policy=session,
        idtoname=idtoname,
        nametoid=nametoid,
        indextoname=indextoname,
        nametoindex=nametoindex,
    )

    # Performance tracking variables
    start_time = time.time()
    end_time = start_time + 10  # Run for 10 seconds like test_00
    frequency = 50  # Hz

    async with KOS(ip=args.host, port=args.port) as sim_kos:
        # Initialize position and orientation
        base_pos = [0.0, 0.0, 1.05]  # x, y, z
        base_quat = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
        
        # Create joint values list in the format expected by sim_pb2.JointValue

        
        default_position = [
            0.0, 0.0, 0.0, -1.5707963267948966, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966, 0.0, 
            0.3490658503988659, 0.0, 0.0, 0.6981317007977318, -0.3490658503988659, -0.3490658503988659, 0.0, 0.0,
            -0.6981317007977318, 0.3490658503988659
        ]
        
        joint_values = []
        for actuator, pos in zip(ACTUATOR_LIST, default_position):
            joint_values.append({"name": actuator.joint_name, "pos": pos})

        # breakpoint()
        await sim_kos.sim.reset(
            pos={"x": base_pos[0], "y": base_pos[1], "z": base_pos[2]},
            quat={"w": base_quat[0], "x": base_quat[1], "y": base_quat[2], "z": base_quat[3]},
            joints=joint_values
        )

        await configure_robot(sim_kos)

        while time.time() < end_time:
            loop_start_time = time.time()
            print(f"Time: {time.time()} end_time: {end_time}")

            if args.sim_only:
                print("Running in simulation-only mode...")
                _ = await robot_state.inner_loop(
                    kos=sim_kos,
                )
                waiting_time = 1 / frequency
                loop_end_time = time.time()
                sleep_time = max(0, waiting_time - (loop_end_time - loop_start_time))
                logger.debug("Sleeping for %s seconds", sleep_time)
                await asyncio.sleep(sleep_time)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--sim-only",
        action="store_true",
        help="Run simulation only without connecting to the real robot",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="assets/saved_checkpoints/2025-02-20_00-28-33_model_2600",
    )
    # default="assets/saved_checkpoints/2025-02-19_02-47-37_model_4250")
    parser.add_argument("--vel_cmd", type=str, default="0.0, 0.0, 0.0")
    args = parser.parse_args()

    args.sim_only = True  # Force sim-only mode to always be True

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    await run_robot(args)


if __name__ == "__main__":
    asyncio.run(main())
