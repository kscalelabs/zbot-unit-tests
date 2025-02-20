"""Run reinforcement learning on the robot simulator."""
import argparse
import asyncio
import logging
import math
import os
import subprocess
import time
from dataclasses import dataclass

import colorlogging
import numpy as np
import onnx
import onnxruntime as ort
import pykos
# from kinfer.inference.python import ONNXModel
from typing import Any
from pykos import KOS
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


@dataclass
class Actuator:
    actuator_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


ACTUATOR_LIST: list[Actuator] = [
    Actuator(actuator_id=11, kp=85.0, kd=2.0, max_torque=60.0, joint_name="left_shoulder_pitch_03"),
    Actuator(actuator_id=12, kp=85.0, kd=2.0, max_torque=60.0, joint_name="left_shoulder_roll_03"),
    Actuator(actuator_id=13, kp=80.0, kd=1.0, max_torque=17.0, joint_name="left_shoulder_yaw_02"),
    Actuator(actuator_id=14, kp=80.0, kd=1.0, max_torque=17.0, joint_name="left_elbow_02"),
    Actuator(actuator_id=15, kp=80.0, kd=1.0, max_torque=17.0, joint_name="left_wrist_02"),

    Actuator(actuator_id=21, kp=85.0, kd=2.0, max_torque=60.0, joint_name="right_shoulder_pitch_03"),
    Actuator(actuator_id=22, kp=85.0, kd=2.0, max_torque=60.0, joint_name="right_shoulder_roll_03"),
    Actuator(actuator_id=23, kp=80.0, kd=1.0, max_torque=17.0, joint_name="right_shoulder_yaw_02"),
    Actuator(actuator_id=24, kp=80.0, kd=1.0, max_torque=17.0, joint_name="right_elbow_02"),
    Actuator(actuator_id=25, kp=80.0, kd=1.0, max_torque=17.0, joint_name="right_wrist_02"),

    Actuator(actuator_id=31, kp=100.0, kd=7.0, max_torque=80.0, joint_name="left_hip_pitch_04"),
    Actuator(actuator_id=32, kp=85.0, kd=2.0, max_torque=60.0, joint_name="left_hip_roll_03"),
    Actuator(actuator_id=33, kp=85.0, kd=2.0, max_torque=60.0, joint_name="left_hip_yaw_03"),
    Actuator(actuator_id=34, kp=100.0, kd=7.0, max_torque=80.0, joint_name="left_knee_04"),
    Actuator(actuator_id=35, kp=80.0, kd=1.0, max_torque=17.0, joint_name="left_ankle_02"),

    Actuator(actuator_id=41, kp=100.0, kd=7.0, max_torque=80.0, joint_name="right_hip_pitch_04"),
    Actuator(actuator_id=42, kp=85.0, kd=2.0, max_torque=60.0, joint_name="right_hip_roll_03"),
    Actuator(actuator_id=43, kp=85.0, kd=2.0, max_torque=60.0, joint_name="right_hip_yaw_03"),
    Actuator(actuator_id=44, kp=100.0, kd=7.0, max_torque=80.0, joint_name="right_knee_04"),
    Actuator(actuator_id=45, kp=80.0, kd=1.0, max_torque=17.0, joint_name="right_ankle_02"),  
]

ACTUATOR_IDS = [actuator.actuator_id for actuator in ACTUATOR_LIST]
# TODO
ACTUATOR_ID_TO_POLICY_IDX = {
    11: 0,  # left_shoulder_pitch_03
    12: 1,  # left_shoulder_roll_03
    13: 2,  # left_shoulder_yaw_02
    14: 3,  # left_elbow_02
    15: 4,  # left_wrist_02
    21: 5,  # right_shoulder_pitch_03
    22: 6,  # right_shoulder_roll_03
    23: 7,  # right_shoulder_yaw_02
    24: 8,  # right_elbow_02
    25: 9,  # right_wrist_02
    31: 10,  # left_hip_pitch_04
    32: 11,  # left_hip_roll_03
    33: 12,  # left_hip_yaw_03
    34: 13,  # left_knee_04
    35: 14,  # left_ankle_02
    41: 15,  # right_hip_pitch_04
    42: 16,  # right_hip_roll_03
    43: 17,  # right_hip_yaw_03
    44: 18,  # right_knee_04
    45: 19,  # right_ankle_02
}


def map_isaac_to_mujoco(isaac_to_mujoco_mapping, isaac_position) -> np.ndarray:
    """Maps joint positions from Isaac format to MuJoCo format.

    Args:
        isaac_position (np.array): Joint positions in Isaac order.

    Returns:
        np.array: Joint positions in MuJoCo order.
    """
    mujoco_position = np.zeros(len(isaac_to_mujoco_mapping))
    for isaac_index, isaac_name in enumerate(isaac_to_mujoco_mapping.keys()):
        mujoco_index = list(isaac_to_mujoco_mapping.values()).index(isaac_name)
        mujoco_position[mujoco_index] = isaac_position[isaac_index]
    return mujoco_position

def map_mujoco_to_isaac(mujoco_to_isaac_mapping, mujoco_position):
    """Maps joint positions from MuJoCo format to Isaac format.

    Args:
        mujoco_position (np.array): Joint positions in MuJoCo order.

    Returns:
        np.array: Joint positions in Isaac order.
    """
    # Create an array for Isaac positions based on the mapping
    isaac_position = np.zeros(len(mujoco_to_isaac_mapping))
    for mujoco_index, mujoco_name in enumerate(mujoco_to_isaac_mapping.keys()):
        isaac_index = list(mujoco_to_isaac_mapping.values()).index(mujoco_name)
        isaac_position[isaac_index] = mujoco_position[mujoco_index]
    return isaac_position


async def simple_walking(
    policy: Any,
    default_position: list[float], 
    host: str, 
    port: int
) -> None:
    async with KOS(ip=host, port=port) as sim_kos:
        # USE CONFIGURE ACTUATOR AT YOUR OWN RISK 

        for actuator in ACTUATOR_LIST: 
            await sim_kos.actuator.configure_actuator(
                actuator_id=actuator.actuator_id,
                kp=actuator.kp,
                kd=actuator.kd,
                max_torque=actuator.max_torque
            )
        # Initialize position and orientation
        base_pos = [0.0, 0.0, 1.05]  # x, y, z
        base_quat = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
        
        # Create joint values list in the format expected by sim_pb2.JointValue
        joint_values = []
        for actuator, pos in zip(ACTUATOR_LIST, default_position):
            joint_values.append({"name": actuator.joint_name, "pos": pos})

        # breakpoint()
        await sim_kos.sim.reset(
            pos={"x": base_pos[0], "y": base_pos[1], "z": base_pos[2]},
            quat={"w": base_quat[0], "x": base_quat[1], "y": base_quat[2], "z": base_quat[3]},
            joints=joint_values
        )



        count = 0
        start_time = time.time()
        end_time = start_time + 10
        last_second = int(time.time())
        second_count = 0

        default = np.array(default_position)
        target_q = np.zeros(20, dtype=np.double)
        last_action = np.zeros(20, dtype=np.double)
        hist_obs = np.zeros(570, dtype=np.double)

        count_lowlevel = 0

        x_vel_cmd = 0.0
        y_vel_cmd = 0.0
        yaw_vel_cmd = -0.1
        frequency = 50

        mujoco_to_isaac_mapping = {
            'left_shoulder_pitch_03': 'left_hip_pitch_04', 
            'left_shoulder_roll_03': 'left_shoulder_pitch_03', 
            'left_shoulder_yaw_02': 'right_hip_pitch_04', 
            'left_elbow_02': 'right_shoulder_pitch_03', 
            'left_wrist_02': 'left_hip_roll_03', 
            'right_shoulder_pitch_03': 'left_shoulder_roll_03', 
            'right_shoulder_roll_03': 'right_hip_roll_03', 
            'right_shoulder_yaw_02': 'right_shoulder_roll_03', 
            'right_elbow_02': 'left_hip_yaw_03', 
            'right_wrist_02': 'left_shoulder_yaw_02', 
            'left_hip_pitch_04': 'right_hip_yaw_03', 
            'left_hip_roll_03': 'right_shoulder_yaw_02', 
            'left_hip_yaw_03': 'left_knee_04', 
            'left_knee_04': 'left_elbow_02', 
            'left_ankle_02': 'right_knee_04', 
            'right_hip_pitch_04': 'right_elbow_02', 
            'right_hip_roll_03': 'left_ankle_02', 
            'right_hip_yaw_03': 'left_wrist_02', 
            'right_knee_04': 'right_ankle_02', 
            'right_ankle_02': 'right_wrist_02'
        }
        isaac_to_mujoco_mapping = {
            'left_hip_pitch_04': 'left_shoulder_pitch_03', 
            'left_shoulder_pitch_03': 'left_shoulder_roll_03', 
            'right_hip_pitch_04': 'left_shoulder_yaw_02', 
            'right_shoulder_pitch_03': 'left_elbow_02', 
            'left_hip_roll_03': 'left_wrist_02', 
            'left_shoulder_roll_03': 'right_shoulder_pitch_03', 
            'right_hip_roll_03': 'right_shoulder_roll_03', 
            'right_shoulder_roll_03': 'right_shoulder_yaw_02', 
            'left_hip_yaw_03': 'right_elbow_02', 
            'left_shoulder_yaw_02': 'right_wrist_02', 
            'right_hip_yaw_03': 'left_hip_pitch_04', 
            'right_shoulder_yaw_02': 'left_hip_roll_03', 
            'left_knee_04': 'left_hip_yaw_03', 
            'left_elbow_02': 'left_knee_04', 
            'right_knee_04': 'left_ankle_02', 
            'right_elbow_02': 'right_hip_pitch_04', 
            'left_ankle_02': 'right_hip_roll_03', 
            'left_wrist_02': 'right_hip_yaw_03', 
            'right_ankle_02': 'right_knee_04', 
            'right_wrist_02': 'right_ankle_02'
        }
        # TODO Copy automatic code

        start_time = time.time()
        while time.time() < end_time:
            loop_start_time = time.time()
        
            response, raw_quat = await asyncio.gather(
                sim_kos.actuator.get_actuators_state(ACTUATOR_IDS),
                sim_kos.imu.get_quaternion()
            )
            positions = np.array([math.radians(state.position) for state in response.states])
            velocities = np.array([math.radians(state.velocity) for state in response.states])
            # r = R.from_quat([raw_quat.x, raw_quat.y, raw_quat.z, raw_quat.w])
            # gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
            projected_gravity = get_gravity_orientation(
                np.array([raw_quat.w, raw_quat.x, raw_quat.y, raw_quat.z])
            )
            cur_pos_obs = positions - default
            cur_vel_obs = velocities

            # 3D velocity commands
            vel_cmd = np.array([x_vel_cmd, y_vel_cmd, yaw_vel_cmd], dtype=np.float32)

            cur_pos_isaac = map_mujoco_to_isaac(mujoco_to_isaac_mapping, cur_pos_obs).astype(np.float32)
            cur_vel_isaac = map_mujoco_to_isaac(mujoco_to_isaac_mapping, cur_vel_obs).astype(np.float32)
            last_act_isaac = last_action.astype(np.float32)

            # Now build the entire 72-D observation:
            #   3 (vel_cmd) + 3 (imu_ang_vel) + 3 (imu_lin_acc) + 3 (proj_grav)
            #   + 20 (joint pos) + 20 (joint vel) + 20 (last_action) = 72
            obs = np.concatenate(
                [
                    vel_cmd,  # 3
                    projected_gravity.astype(np.float32),  # 3
                    cur_pos_isaac,  # 20
                    cur_vel_isaac,  # 20
                    last_act_isaac,  # 20
                ],
            )
            input_name = policy.get_inputs()[0].name
            curr_actions = policy.run(None, {input_name: obs.reshape(1, -1).astype(np.float32)})[0][0]
            last_action = curr_actions.copy()

            curr_actions_scaled = curr_actions * 0.5
            target_q = map_isaac_to_mujoco(isaac_to_mujoco_mapping, curr_actions_scaled)
            target_q += default

            commands = []
            for actuator_id in ACTUATOR_IDS:
                policy_idx = ACTUATOR_ID_TO_POLICY_IDX[actuator_id]
                raw_value = target_q[policy_idx]
                command_deg = raw_value
                command_deg = math.degrees(raw_value)
                commands.append({"actuator_id": actuator_id, "position": command_deg})
            
            await sim_kos.actuator.command_actuators(commands)
            print(commands)
            waiting_time = 1 / frequency
            loop_end_time = time.time()
            sleep_time = max(0, waiting_time - (loop_end_time - loop_start_time))
            await asyncio.sleep(sleep_time)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sim-only", action="store_true",
                        help="Run simulation only without connecting to the real robot")
    parser.add_argument("--model-path", 
        type=str, 
        default="assets/saved_checkpoints/2025-02-20_00-28-33_model_2600",
        help="Path to the model to simulate"
    )
    args = parser.parse_args()
 
    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    # Get the most recent yaml and onnx files from the checkpoint directory
    yaml_files = [f for f in os.listdir(args.model_path) if f.endswith('env.yaml')]
    policy_files = [f for f in os.listdir(args.model_path) if f.endswith('.onnx')]
    
    if not yaml_files or not policy_files:
        raise FileNotFoundError(f"Could not find env.yaml and .onnx files in {args.model_path}")

    yaml_file = yaml_files[0]  # Use first found yaml
    policy_file = policy_files[0]  # Use first found onnx
    policy_path = os.path.join(args.model_path, policy_file)
    yaml_path = os.path.join(args.model_path, yaml_file)

    try:
        print("Running in simulation-only mode...")

        policy = onnx.load(policy_path)
        session = ort.InferenceSession(policy.SerializeToString())

        default_position = [
            0.0, 0.0, 0.0, -1.5707963267948966, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966, 0.0, 
            0.3490658503988659, 0.0, 0.0, 0.6981317007977318, -0.3490658503988659, -0.3490658503988659, 0.0, 0.0,
            -0.6981317007977318, 0.3490658503988659
        ]

        await simple_walking(session, default_position, args.host, args.port)

    except Exception:
        logger.exception("Simulator error")
        raise

if __name__ == "__main__":
    asyncio.run(main())