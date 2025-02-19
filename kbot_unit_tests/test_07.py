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
        vel_cmd: np.ndarray
    ):
        self.joint_offsets = {name: 0.0 for name in joint_names}
        self.joint_signs = joint_signs
        self.orn_offset = None
        self.start_pos = start_pos
        self.vel_cmd = vel_cmd
        
    async def offset_in_place(self, kos: KOS, joint_names: list[str] | None = None) -> None:
        """Capture current position as zero offset."""
        # Get current joint positions (in degrees)
        # states = await kos.actuator.get_actuators_state([JOINT_NAME_TO_ID[name] for name in joint_names])
        # current_positions = {name: states.states[i].position for i, name in enumerate(joint_names)}

        # # Store negative of current positions as offsets (in degrees)
        # self.joint_offsets = {name: 0.0 for name, _ in current_positions.items()}

        # Store IMU offset
        imu_data = await kos.imu.get_euler_angles()
        initial_quat = R.from_euler("xyz", [imu_data.roll, imu_data.pitch, imu_data.yaw], degrees=True).as_quat()
        self.orn_offset = R.from_quat(initial_quat).inv()
        
    def map_isaac_to_kos_sim(self, q: np.ndarray) -> np.ndarray:
        
        for i, value in enumerate(q):
            q[i] = value
        return q

    def map_kos_sim_to_isaac(self, states: list[Any], aid_to_jname: dict[int, str], pi_to_jname: dict[int, str], jname_to_pi: dict[str, int], pos = True) -> np.ndarray:
        vec = np.zeros(len(isaac_joint_names))
        for i in range(len(isaac_joint_names)):
            curr_id = states.states[i].actuator_id
            curr_name = aid_to_jname[curr_id]
            curr_value = states.states[i].position if pos else states.states[i].velocity
            
            vec[jname_to_pi[curr_name]] = np.deg2rad(curr_value - self.start_pos[curr_name]) * self.joint_signs[curr_name]
            
            logger.debug(f" i: {i}, id: {curr_id}, name: {curr_name}, {'pos' if pos else 'vel'} value: {curr_value}")
        return vec
    
    def map_isaac_to_kos_sim(self, actions: np.ndarray, jname_to_aid: dict[str, int], pi_to_jname: dict[int, str]) -> list[dict]:
        commands = []
        for i in range(len(actions)):
            target_pos = np.rad2deg(actions[i] + self.start_pos[pi_to_jname[i]]) * self.joint_signs[pi_to_jname[i]]
            commands.append({"actuator_id": jname_to_aid[pi_to_jname[i]], "position": target_pos})
        return commands

    async def get_obs(self, kos: KOS, jname_to_aid: dict[str, int], aid_to_jname: dict[int, str], pi_to_jname: dict[int, str], jname_to_pi: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get robot state with offset compensation."""
        # Batch state requests
        states, euler_data, imu_sensor_data = await asyncio.gather(
            kos.actuator.get_actuators_state([actuator.actuator_id for actuator in ACTUATOR_LIST]),
            kos.imu.get_euler_angles(),
            kos.imu.get_imu_values(),
        )

        # Apply offsets and signs to positions and convert to radians
        
        q = self.map_kos_sim_to_isaac(states, aid_to_jname, pi_to_jname, jname_to_pi, pos = True)
        dq = self.map_kos_sim_to_isaac(states, aid_to_jname, pi_to_jname, jname_to_pi, pos = False)
            
            
            
            
            
        # q = np.array(
        #     [
        #         np.deg2rad((states.states[i].position - self.start_pos[aid_to_jname[actuator.actuator_id]]) * self.joint_signs[aid_to_jname[actuator.actuator_id]])
        #         for i, actuator in enumerate(ACTUATOR_LIST)
        #     ],
        #     dtype=np.float32,
        # )
        
        # q = self.map_isaac_to_kos_sim(q)

        # for i, name in enumerate(isaac_joint_names):
            

        # Apply signs to velocities and convert to radians
        # dq = np.array(
        #     [
        #         np.deg2rad(states.states[i].velocity * self.joint_signs[aid_to_jname[actuator.actuator_id]])
        #         for i, actuator in enumerate(ACTUATOR_LIST)
        #     ],
        #     dtype=np.float32,
        # )

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

        # omega = np.zeros(3)
        # quat = np.array([1.0, 0.0, 0.0, 0.0])

        return q, dq, quat, gvec, omega, self.vel_cmd
    
    def apply_command(self, position: float, joint_name: str) -> float:
        """Apply sign first, then offset to outgoing command. Convert from radians to degrees."""
        position_deg = np.rad2deg(position)
        return position_deg * self.joint_signs[joint_name] - self.joint_offsets[joint_name]


async def inner_loop(
    kos: KOS,
    robot_state: RobotState,
    session: ort.InferenceSession,
    config: dict,
    aid_to_jname: dict,
    jname_to_aid: dict,
    pi_to_jname: dict,
    jname_to_pi: dict,
    prev_action: np.ndarray,
    count: int,
    start_time: float,
    end_time: float,
    last_second: int,
    second_count: int,
) -> None:
    
    q, dq, quat, gvec, omega, vel_cmd = await robot_state.get_obs(kos, jname_to_aid, aid_to_jname, pi_to_jname, jname_to_pi)

    logger.debug(f"q: {q}")
    logger.debug(f"dq: {dq}")
    logger.debug(f"quat: {quat}")
    logger.debug(f"gvec: {gvec}")
    logger.debug(f"omega: {omega}")
    
    obs = np.concatenate([
        vel_cmd,
        gvec,
        q,
        dq,
        prev_action
    ])

    input_name = session.get_inputs()[0].name
    curr_actions_raw = session.run(
        None, {input_name: obs.reshape(1, -1).astype(np.float32)}
    )[0][0]  # shape (20,)
    
    curr_actions = curr_actions_raw * config["actions"]["joint_pos"]["scale"]
    logger.debug(f"curr_actions: {curr_actions}")
    
    # curr_actions = robot_state.map_isaac_to_kos_sim(curr_actions)
    
    commands = robot_state.map_isaac_to_kos_sim(curr_actions, jname_to_aid, pi_to_jname)

    # robot_actions = {}
    # commands = []
    
    # for i, joint_name in enumerate(isaac_joint_names):
    #     robot_actions[jname_to_aid[joint_name]] = robot_state.apply_command(curr_actions[jname_to_pi[joint_name]], joint_name)
    #     commands.append({"actuator_id": jname_to_aid[joint_name], "position": robot_actions[jname_to_aid[joint_name]]})
        
    
    
    logger.debug("Commands:")
    for cmd in commands:
        logger.debug(f"  {cmd['actuator_id']}: {cmd['position']:.2f} deg")
    
    logger.debug("Sending commands to actuators")
    await kos.actuator.command_actuators(commands)
    
    return curr_actions_raw
    
    # states, euler_data, imu_sensor_data = await asyncio.gather(
    # kos.actuator.get_actuators_state([actuator.actuator_id for actuator in ACTUATOR_LIST]),
    # kos.imu.get_euler_angles(),
    # kos.imu.get_imu_values(),
    # )
    
    # logger.debug(f"States: {states}")
    # logger.debug(f"Euler data: {euler_data}")
    # logger.debug(f"IMU sensor data: {imu_sensor_data}")


async def run_robot(args: argparse.Namespace) -> None:
    
    session, config = await load_policy_and_config(args.checkpoint_path)
    
    # Create mappings between actuator ID and joint name
    aid_to_jname = {actuator.actuator_id: actuator.joint_name for actuator in ACTUATOR_LIST}
    jname_to_aid = {actuator.joint_name: actuator.actuator_id for actuator in ACTUATOR_LIST}

    # Example policy joint mapping (assumes policy outputs match the order of joints)
    # Assuming your policy output corresponds to the joints in a specific order (as per ACTUATOR_LIST)
    pi_to_jname = {i: isaac_joint_names[i] for i in range(len(isaac_joint_names))}
    jname_to_pi = {isaac_joint_names[i]: i for i in range(len(isaac_joint_names))}
    
    vel_cmd = np.array(args.vel_cmd.split(","), dtype=np.float32)
    robot_state = RobotState(isaac_joint_names, joint_signs, start_pos, vel_cmd)

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
            pass
        
            if args.sim_only:
                print("Running in simulation-only mode...")
                curr_actions_raw = await inner_loop(
                    kos = sim_kos,
                    robot_state = robot_state,
                    session = session,
                    config = config,
                    aid_to_jname = aid_to_jname,
                    jname_to_aid = jname_to_aid,
                    pi_to_jname = pi_to_jname,
                    jname_to_pi = jname_to_pi,
                    prev_action = prev_action,
                    count = count,
                    start_time = start_time,
                    end_time = end_time,
                    last_second = last_second,
                    second_count = second_count
                )
                prev_action = curr_actions_raw
                await asyncio.sleep(0.1)
                # print("Homing...")
                # homing_command = [{
                #     "actuator_id": actuator.actuator_id,
                #     "position": 0.0
                # } for actuator in ACTUATOR_LIST]
                
                # await sim_kos.actuator.command_actuators(homing_command)
                # await asyncio.sleep(3)

                # order = [11, 21, 12, 22, 13, 23, 14, 24, 15, 25, 31, 41, 32, 42, 33, 43, 34, 44, 35, 45]

                # actuator_map = {actuator.actuator_id: actuator for actuator in ACTUATOR_LIST}
                
                # for actuator_id in order:
                #     actuator = actuator_map.get(actuator_id)
                #     print(f"Testing {actuator.actuator_id} in simulation...")

                #     TEST_ANGLE = -10.0

                #     FLIP_SIGN = 1.0 if actuator.actuator_id in [12, 21, 13, 14, 15, 25, 32, 33, 35, 41, 44] else -1.0
                #     TEST_ANGLE *= FLIP_SIGN

                #     command = [{
                #         "actuator_id": actuator.actuator_id,
                #         "position": TEST_ANGLE,
                #     }]
                #     print(f"Sending command to actuator {actuator.actuator_id}: name {actuator.joint_name} position {TEST_ANGLE}")

                #     await sim_kos.actuator.command_actuators(command)
                #     await asyncio.sleep(2)

                #     command = [{
                #         "actuator_id": actuator.actuator_id,
                #         "position": 0.0,
                #     }]
                #     await sim_kos.actuator.command_actuators(command)
                # await asyncio.sleep(0.1)
        # else:
        #     print("Running on both simulator and real robots simultaneously...")
        #     async with KOS(ip=args.host, port=args.port) as sim_kos, \
        #                KOS(ip="100.89.14.31", port=args.port) as real_kos:
        #         await sim_kos.sim.reset()
        #         await configure_robot(sim_kos)
        #         await configure_robot(real_kos)
        #         print("Homing...")
        #         homing_command = [{
        #             "actuator_id": actuator.actuator_id,
        #             "position": 0.0
        #         } for actuator in ACTUATOR_LIST]
        #         await asyncio.gather(
        #             sim_kos.actuator.command_actuators(homing_command),
        #             real_kos.actuator.command_actuators(homing_command),
        #         )
        #         await asyncio.sleep(2)


        #         order = [11, 21, 12, 22, 13, 23, 14, 24, 15, 25, 31, 41, 32, 42, 33, 43, 34, 44, 35, 45]

        #         actuator_map = {actuator.actuator_id: actuator for actuator in ACTUATOR_LIST}

        #         for actuator_id in order:
        #                 actuator = actuator_map.get(actuator_id)
        #                 print(f"Testing {actuator.actuator_id} in simulation...")

        #                 TEST_ANGLE = -10.0

        #                 FLIP_SIGN = 1.0 if actuator.actuator_id in [12, 21, 13, 14, 15, 25, 32, 33, 35, 41, 44] else -1.0
        #                 TEST_ANGLE *= FLIP_SIGN

        #                 command = [{
        #                     "actuator_id": actuator.actuator_id,
        #                     "position": TEST_ANGLE,
        #                 }]
        #                 print(f"Sending command to actuator {actuator.actuator_id}: name {actuator.joint_name} position {TEST_ANGLE}")

        #                 await asyncio.gather(
        #                     sim_kos.actuator.command_actuators(command),
        #                     real_kos.actuator.command_actuators(command),
        #                 )
        #                 await asyncio.sleep(2)

        #                 command = [{
        #                     "actuator_id": actuator.actuator_id,
        #                     "position": 0.0,
        #                 }]
        #                 await asyncio.gather(
        #                     sim_kos.actuator.command_actuators(command),
        #                     real_kos.actuator.command_actuators(command),
        #                 )
                        
        #                 await asyncio.sleep(2)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sim-only", action="store_true",
                        help="Run simulation only without connecting to the real robot")
    parser.add_argument("--checkpoint-path", type=str, 
                        default="assets/saved_checkpoints/2025-02-19_02-47-37_model_4250")
    parser.add_argument("--vel_cmd", type=str, default="0.2, 0.2, 0.2")
    args = parser.parse_args()
    
    args.sim_only = True  # Force sim-only mode to always be True
 
    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)
    


    sim_process = subprocess.Popen(["kos-sim", "kbot-v1"])
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
