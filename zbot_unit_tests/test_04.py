"""Runs reinforcement learning unit tests.

To see a video of the policy running in simulation, look in `assets/model_checkpoints/zbot_rl_policy/policy.mp4`.
"""

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
    "left_hip_yaw",
    "left_shoulder_yaw",
    "right_hip_yaw",
    "right_shoulder_yaw",
    "left_hip_roll",
    "left_shoulder_pitch",
    "right_hip_roll",
    "right_shoulder_pitch",
    "left_hip_pitch",
    "left_elbow",
    "right_hip_pitch",
    "right_elbow",
    "left_knee",
    "left_gripper",
    "right_knee",
    "right_gripper",
    "left_ankle",
    "right_ankle",
]

start_pos = {
    "left_hip_yaw": 0.0,
    "left_shoulder_yaw": 0.0,
    "right_hip_yaw": 0.0,
    "right_shoulder_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_shoulder_pitch": 0.0,
    "right_hip_roll": 0.0,
    "right_shoulder_pitch": 0.0,
    "left_hip_pitch": -math.radians(31.6),
    "left_elbow": 0.0,
    "right_hip_pitch": math.radians(31.6),
    "right_elbow": 0.0,
    "left_knee": math.radians(65.6),
    "left_gripper": 0.0,
    "right_knee": -math.radians(65.6),
    "right_gripper": 0.0,
    "left_ankle": math.radians(31.6),
    "right_ankle": -math.radians(31.6),
}

# start_pos = {
#     "left_hip_yaw": 0.0,
#     "left_shoulder_yaw": 0.0,
#     "right_hip_yaw": 0.0,
#     "right_shoulder_yaw": 0.0,
#     "left_hip_roll": 0.0,
#     "left_shoulder_pitch": 0.0,
#     "right_hip_roll": 0.0,
#     "right_shoulder_pitch": 0.0,
#     "left_hip_pitch": 0.0,
#     "left_elbow": 0.0,
#     "right_hip_pitch": 0.0,
#     "right_elbow": 0.0,
#     "left_knee": 0.0,
#     "left_gripper": 0.0,
#     "right_knee": 0.0,
#     "right_gripper": 0.0,
#     "left_ankle": 0.0,
#     "right_ankle": 0.0,
# }

joint_inversions = {
    "left_hip_yaw": 1,
    "left_shoulder_yaw": 1,
    "right_hip_yaw": 1,
    "right_shoulder_yaw": 1,
    "left_hip_roll": 1,
    "left_shoulder_pitch": 1,
    "right_hip_roll": 1,
    "right_shoulder_pitch": 1,
    "left_hip_pitch": 1,
    "left_elbow": 1,
    "right_hip_pitch": 1,
    "right_elbow": 1,
    "left_knee": 1,
    "left_gripper": 1,
    "right_knee": -1,
    "right_gripper": 1,
    "left_ankle": 1,
    "right_ankle": 1,
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
    Actuator(actuator_id=31, nn_id=1, kp=150.0, kd=5.084, max_torque=2.90, joint_name="left_hip_yaw"),
    Actuator(actuator_id=11, nn_id=2, kp=150.0, kd=5.084, max_torque=2.90, joint_name="left_shoulder_yaw"),
    Actuator(actuator_id=41, nn_id=3, kp=150.0, kd=5.084, max_torque=2.90, joint_name="right_hip_yaw"),
    Actuator(actuator_id=21, nn_id=4, kp=150.0, kd=5.084, max_torque=2.90, joint_name="right_shoulder_yaw"),
    Actuator(actuator_id=32, nn_id=5, kp=150.0, kd=5.084, max_torque=2.90, joint_name="left_hip_roll"),
    Actuator(actuator_id=12, nn_id=6, kp=150.0, kd=5.084, max_torque=2.90, joint_name="left_shoulder_pitch"),
    Actuator(actuator_id=42, nn_id=7, kp=150.0, kd=5.084, max_torque=2.90, joint_name="right_hip_roll"),
    Actuator(actuator_id=22, nn_id=8, kp=150.0, kd=5.084, max_torque=2.90, joint_name="right_shoulder_pitch"),
    Actuator(actuator_id=33, nn_id=9, kp=150.0, kd=5.084, max_torque=2.90, joint_name="left_hip_pitch"),
    Actuator(actuator_id=13, nn_id=10, kp=150.0, kd=5.084, max_torque=2.90, joint_name="left_elbow"),
    Actuator(actuator_id=43, nn_id=11, kp=150.0, kd=5.084, max_torque=2.90, joint_name="right_hip_pitch"),
    Actuator(actuator_id=23, nn_id=12, kp=150.0, kd=5.084, max_torque=2.90, joint_name="right_elbow"),
    Actuator(actuator_id=34, nn_id=13, kp=150.0, kd=5.084, max_torque=2.90, joint_name="left_knee"),
    Actuator(actuator_id=14, nn_id=14, kp=150.0, kd=5.084, max_torque=2.90, joint_name="left_gripper"),
    Actuator(actuator_id=44, nn_id=15, kp=150.0, kd=5.084, max_torque=2.90, joint_name="right_knee"),
    Actuator(actuator_id=24, nn_id=16, kp=150.0, kd=5.084, max_torque=2.90, joint_name="right_gripper"),
    Actuator(actuator_id=35, nn_id=17, kp=150.0, kd=5.084, max_torque=2.90, joint_name="left_ankle"),
    Actuator(actuator_id=45, nn_id=18, kp=150.0, kd=5.084, max_torque=2.90, joint_name="right_ankle"),
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
    ) -> None:
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
        gvec = r.apply(np.array([-0.01298109, -0.27928606, -0.93012038]), inverse=True)

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
            curr_value = joint_inversions[curr_name] * curr_value
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

    def map_isaac_to_kos_sim(self, actions: np.ndarray) -> list[dict]:
        """Map Isaac state to KOS state."""
        commands = []

        for i in range(len(actions)):
            curr_name = self.indextoname[i]
            curr_id = self.nametoid[curr_name]
            target_pos = np.rad2deg(actions[i] + self.start_pos[curr_name])
            commands.append({"actuator_id": curr_id, "position": target_pos})
        return commands

    async def gather_kos_data(self, kos: KOS) -> tuple[list[Any], Any, Any]:
        """Gather data from KOS."""
        states = await kos.actuator.get_actuators_state([actuator.actuator_id for actuator in ACTUATOR_LIST])
        # euler_data = await kos.imu.get_euler_angles()
        # imu_data = await kos.imu.get_imu_values()
        euler_data = None
        imu_data = None
        return states, euler_data, imu_data

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

        # gvec, quat, omega = self.get_gravity_vector(euler_data, imu_sensor_data)

        gvec = np.array([0.01029401, -0.26815441, -0.96332127])

        logger.debug("vel_cmd: %s", self.vel_cmd)
        logger.debug("gvec: %s", gvec)
        logger.debug("q: %s", q)
        logger.debug("dq: %s", dq)
        logger.debug("prev_action: %s", prev_action)

        logger.debug("Shape of vel_cmd: %s", self.vel_cmd.shape)
        logger.debug("Shape of gvec: %s", gvec.shape)
        logger.debug("Shape of q: %s", q.shape)
        logger.debug("Shape of dq: %s", dq.shape)
        logger.debug("Shape of prev_action: %s", prev_action.shape)

        obs = np.concatenate([self.vel_cmd, gvec, q, dq, prev_action])

        return obs

    def apply_command(self, position: float, joint_name: str) -> float:
        """Apply sign first, then offset to outgoing command. Convert from radians to degrees."""
        position_deg = np.rad2deg(position)
        return position_deg - self.joint_offsets[joint_name]

    async def inner_loop(self, kos: KOS) -> None:
        states, euler_data, imu_sensor_data = await self.gather_kos_data(kos)
        obs = self.get_obs(states, euler_data, imu_sensor_data, self.prev_action)
        input_name = self.policy.get_inputs()[0].name
        actions_raw = self.policy.run(None, {input_name: obs.reshape(1, -1).astype(np.float32)})[0][0]  # shape (20,)

        # actions_raw = np.zeros_like(actions_raw) # ZERO ACTIONS
        self.prev_action = actions_raw
        actions = actions_raw * self.config["actions"]["joint_pos"]["scale"]
        logger.debug("actions: %s", actions)

        commands = self.map_isaac_to_kos_sim(actions)
        for cmd in commands:
            logger.debug(
                "cmd id: %s, pos: %s deg, name: %s",
                cmd["actuator_id"],
                cmd["position"],
                self.idtoname[cmd["actuator_id"]],
            )

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
        # # Initialize position and orientation
        base_pos = [0.0, 0.0, 1.05]  # x, y, z
        base_quat = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z

        # Create joint values list in the format expected by sim_pb2.JointValue
        joint_values = []
        # for actuator, pos in zip(ACTUATOR_LIST, [start_pos[name] for name in isaac_joint_names]):
        #     value = pos
        #     logger.debug("appending actuator.joint_name: %s, value: %s", actuator.joint_name, value)
        #     joint_values.append({"name": actuator.joint_name, "pos": value})

        await sim_kos.sim.reset(
            pos={"x": base_pos[0], "y": base_pos[1], "z": base_pos[2]},
            quat={"w": base_quat[0], "x": base_quat[1], "y": base_quat[2], "z": base_quat[3]},
            joints=joint_values,
        )

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
        default="assets/model_checkpoints/2025-02-14_19-18-23_model_13550",
    )
    # default="assets/saved_checkpoints/2025-02-19_02-47-37_model_4250")
    parser.add_argument("--vel_cmd", type=str, default="0.0, -0.5, 0.0")
    args = parser.parse_args()

    args.sim_only = True  # Force sim-only mode to always be True

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    await run_robot(args)


if __name__ == "__main__":
    asyncio.run(main())
