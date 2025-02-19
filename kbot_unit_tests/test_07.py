import argparse
import asyncio
import logging
import os
import yaml
import subprocess
import time
from dataclasses import dataclass

import colorlogging
from pykos import KOS
from typing import Any
from scipy.spatial.transform import Rotation as R

import onnx
import onnxruntime as ort
import numpy as np
import math

logger = logging.getLogger(__name__)


# New function to load and prepare the policy
async def load_policy_and_config(checkpoint_path: str):
    try:
        # List YAML and ONNX files in the given directory
        yaml_files = [f for f in os.listdir(checkpoint_path) if f.endswith('env.yaml')]
        policy_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.onnx')]

        if not yaml_files or not policy_files:
            raise FileNotFoundError(f"Could not find env.yaml and .onnx files in {checkpoint_path}")

        yaml_file = yaml_files[0]  # Use the first found YAML file
        policy_file = policy_files[0]  # Use the first found ONNX file

        policy_path = os.path.join(checkpoint_path, policy_file)
        yaml_path = os.path.join(checkpoint_path, yaml_file)

        logger.info(f"Loading policy from: {os.path.abspath(policy_path)}")
        logger.info(f"Loading config from: {os.path.abspath(yaml_path)}")

        # Load the policy and create an inference session
        policy = onnx.load(policy_path)
        session = ort.InferenceSession(policy.SerializeToString())
        
        with open(yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        
        return session, config
    except Exception as e:
        logger.error(f"Error in load_policy_and_config: {e}")
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
    "right_wrist_02"
]

joint_signs = {
    "left_hip_pitch_04": 1, 
    "left_shoulder_pitch_03": 1, 
    "right_hip_pitch_04": 1, 
    "right_shoulder_pitch_03": 1, 
    "left_hip_roll_03": 1, 
    "left_shoulder_roll_03": 1, 
    "right_hip_roll_03": 1, 
    "right_shoulder_roll_03": 1, 
    "left_hip_yaw_03": 1, 
    "left_shoulder_yaw_02": 1, 
    "right_hip_yaw_03": 1, 
    "right_shoulder_yaw_02": 1, 
    "left_knee_04": 1,
    "left_elbow_02": 1, 
    "right_knee_04": 1, 
    "right_elbow_02": 1, 
    "left_ankle_02": 1, 
    "left_wrist_02": 1, 
    "right_ankle_02": 1, 
    "right_wrist_02": 1
}
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
    "left_hip_pitch_04": math.radians(20),      # Slight forward lean
    "left_hip_roll_03": 0.0,       # Neutral stance
    "left_hip_yaw_03": 0.0,
    "left_knee_04": math.radians(40),           # Bent knees for stability
    "left_ankle_02": -math.radians(20),         # Compensate for knee bend
    "right_hip_pitch_04": -math.radians(20),
    "right_hip_roll_03": 0.0,
    "right_hip_yaw_03": 0.0,
    "right_knee_04": -math.radians(40),
    "right_ankle_02": math.radians(20)
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
    Actuator(13, 9, 30.0, 1.0, 17.0, "left_shoulder_yaw_02"),
    Actuator(14, 13, 30.0, 1.0, 17.0, "left_elbow_02"),
    Actuator(15, 17, 30.0, 1.0, 17.0, "left_wrist_02"),

    Actuator(21, 3, 85.0, 2.0, 60.0, "right_shoulder_pitch_03"),
    Actuator(22, 7, 85.0, 2.0, 60.0, "right_shoulder_roll_03"),
    Actuator(23, 11, 30.0, 1.0, 17.0, "right_shoulder_yaw_02"),
    Actuator(24, 15, 30.0, 1.0, 17.0, "right_elbow_02"),
    Actuator(25, 19, 30.0, 1.0, 17.0, "right_wrist_02"),

    Actuator(31, 0, 85.0, 3.0, 80.0, "left_hip_pitch_04"),
    Actuator(32, 4, 85.0, 2.0, 60.0, "left_hip_roll_03"),
    Actuator(33, 8, 30.0, 2.0, 60.0, "left_hip_yaw_03"),
    Actuator(34, 12, 60.0, 2.0, 80.0, "left_knee_04"),
    Actuator(35, 16, 80.0, 1.0, 17.0, "left_ankle_02"),

    Actuator(41, 2, 85.0, 3.0, 80.0, "right_hip_pitch_04"),
    Actuator(42, 6, 85.0, 2.0, 60.0, "right_hip_roll_03"),
    Actuator(43, 10, 30.0, 2.0, 60.0, "right_hip_yaw_03"),
    Actuator(44, 14, 60.0, 2.0, 80.0, "right_knee_04"),
    Actuator(45, 18, 80.0, 1.0, 17.0, "right_ankle_02"),  
]

async def configure_robot(kos: KOS) -> None:
    for actuator in ACTUATOR_LIST:
        await kos.actuator.configure_actuator(
            actuator_id = actuator.actuator_id,
            kp = actuator.kp,
            kd = actuator.kd,
            max_torque = actuator.max_torque,
            torque_enabled = True
        )

class RobotState:
    """Tracks robot state and handles offsets."""

    def __init__(
        self, 
        joint_names: list[str], 
        joint_signs: dict[str, float], 
        start_pos: dict[str, float], 
        vel_cmd: np.ndarray,
        config: dict,
        policy: ort.InferenceSession,
        idtoname: dict[int, str],
        nametoid: dict[str, int],
        indextoname: dict[int, str],
        nametoindex: dict[str, int]
    ):
        self.joint_offsets = {name: 0.0 for name in joint_names}
        self.joint_signs = joint_signs
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


    def map_kos_sim_to_isaac(self, states: list[Any], pos = True) -> np.ndarray:
        """Map KOS state to Isaac state."""

        result = np.zeros(len(isaac_joint_names))

        for i in range(len(isaac_joint_names)):

            curr_id = states.states[i].actuator_id
            curr_name = self.idtoname[curr_id]
            curr_value = states.states[i].position if pos else states.states[i].velocity
            curr_value_rad = np.deg2rad(curr_value)
            
            result[self.nametoindex[curr_name]] = (curr_value_rad - self.start_pos[curr_name]) * self.joint_signs[curr_name]
            
            logger.debug(f"{'pos' if pos else 'vel'} obs i: {i}, id: {curr_id}, name: {curr_name},  value: {curr_value}")
        return result
    

    def map_isaac_to_kos_sim(self, actions: np.ndarray) -> list[dict]:
        """Map Isaac state to KOS state."""

        commands = []

        for i in range(len(actions)):
            curr_name = self.indextoname[i]
            curr_id = self.nametoid[curr_name]
            target_pos = np.rad2deg(actions[i] + self.start_pos[curr_name]) * self.joint_signs[curr_name]
            commands.append({"actuator_id": curr_id, "position": target_pos})
        return commands


    async def get_obs(self, kos: KOS, prev_action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get robot state with offset compensation."""
        # Batch state requests
        states, euler_data, imu_sensor_data = await asyncio.gather(
            kos.actuator.get_actuators_state([actuator.actuator_id for actuator in ACTUATOR_LIST]),
            kos.imu.get_euler_angles(),
            kos.imu.get_imu_values(),
        )
        
        q = self.map_kos_sim_to_isaac(states, pos = True)
        dq = self.map_kos_sim_to_isaac(states, pos = False)

        gvec, quat, omega = self.get_gravity_vector(euler_data, imu_sensor_data)

        logger.debug(f"vel_cmd: {self.vel_cmd}")
        logger.debug(f"gvec: {gvec}")
        logger.debug(f"q: {q}")
        logger.debug(f"dq: {dq}")
        logger.debug(f"prev_action: {prev_action}")
        
        obs = np.concatenate([
            self.vel_cmd,
            gvec,
            q,
            dq,
            prev_action
        ])

        return obs
    
    def apply_command(self, position: float, joint_name: str) -> float:
        """Apply sign first, then offset to outgoing command. Convert from radians to degrees."""
        position_deg = np.rad2deg(position)
        return position_deg * self.joint_signs[joint_name] - self.joint_offsets[joint_name]


    async def inner_loop(
        self,
        kos: KOS,
    ) -> None:
        
        obs = await self.get_obs(kos, self.prev_action)

        input_name = self.policy.get_inputs()[0].name
        actions_raw = self.policy.run(
            None, {input_name: obs.reshape(1, -1).astype(np.float32)}
        )[0][0]  # shape (20,)
        self.prev_action = actions_raw
        
        actions = actions_raw * self.config["actions"]["joint_pos"]["scale"]

        logger.debug(f"actions: {actions}")
        
        commands = self.map_isaac_to_kos_sim(actions)
        [logger.debug(f"  {cmd['actuator_id']}: {cmd['position']:.2f} deg") for cmd in commands]
        
        await kos.actuator.command_actuators(commands)
        
        return None


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
        joint_signs=joint_signs,
        start_pos=start_pos,
        vel_cmd=vel_cmd,
        config=config,
        policy=session,
        idtoname=idtoname,
        nametoid=nametoid,
        indextoname=indextoname,
        nametoindex=nametoindex,
    )

    # logger.debug(config)
    # logger.debug(session)

    # Initialize previous actions
    prev_action = np.zeros(20, dtype=np.float32)

    # Performance tracking variables
    count = 0
    start_time = time.time()
    end_time = start_time + 10  # Run for 10 seconds like test_00

    last_second = int(time.time())
    second_count = 0

    async with KOS(ip=args.host, port=args.port) as sim_kos:

        await sim_kos.sim.reset()
        await configure_robot(sim_kos)
        await robot_state.offset_in_place(sim_kos)

        while time.time() < end_time:
            print(f"Time: {time.time()} end_time: {end_time}")

            if args.sim_only:
                print("Running in simulation-only mode...")
                _ = await robot_state.inner_loop(
                    kos = sim_kos,
                )
                await asyncio.sleep(0.01)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sim-only", action="store_true",
                        help="Run simulation only without connecting to the real robot")
    parser.add_argument("--checkpoint-path", type=str, 
                        default="assets/saved_checkpoints/2025-02-19_21-41-28_model_3400")
                        # default="assets/saved_checkpoints/2025-02-19_02-47-37_model_4250")
    parser.add_argument("--vel_cmd", type=str, default="0.2, 0.2, 0.2")
    args = parser.parse_args()
    
    args.sim_only = True  # Force sim-only mode to always be True
 
    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)
    sim_process = subprocess.Popen(["kos-sim", "kbot-v1-feet"])
    time.sleep(2)
    try:
        await run_robot(args)

    except Exception as e:
        print(f"Error: {e}")
        raise e

    finally:
        sim_process.terminate()
        sim_process.wait()


if __name__ == "__main__":
    asyncio.run(main())
