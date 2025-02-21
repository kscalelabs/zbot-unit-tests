"""Mujoco validaiton."""

import argparse
import asyncio
import itertools
import logging
import math
import os
import time
from typing import Dict, Union

import cv2
import mujoco
import mujoco_viewer
import numpy as np
import onnx
import onnxruntime as ort
import yaml
from kos_sim.utils import get_sim_artifacts_path
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput
from mujoco_scenes.mjcf import load_mjmodel
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def get_model_metadata(api: K, model_name: str) -> RobotURDFMetadataOutput:
    model_path = get_sim_artifacts_path() / model_name / "metadata.json"
    if model_path.exists():
        return RobotURDFMetadataOutput.model_validate_json(model_path.read_text())
    model_path.parent.mkdir(parents=True, exist_ok=True)
    robot_class = await api.get_robot_class(model_name)
    metadata = robot_class.metadata
    if metadata is None:
        raise ValueError(f"No metadata found for model {model_name}")
    model_path.write_text(metadata.model_dump_json())
    return metadata


async def get_model_path_and_metadata(model_name: str) -> None:
    async with K() as api:
        model_dir, model_metadata = await asyncio.gather(
            api.download_and_extract_urdf(model_name),
            get_model_metadata(api, model_name),
        )

    model_path = next(
        itertools.chain(
            model_dir.glob("*.mjcf"),
            model_dir.glob("*.xml"),
        )
    )

    return model_path, model_metadata


def save_video_cv2(
    frames: Union[np.ndarray, list[np.ndarray]],
    output_path: str,
    fps: int = 30,
    resize_factor: float = 1.0,
) -> None:  # 1.0 means no resize, 0.5 means half size
    """Save video frames using OpenCV - with compression options.

    Args:
        frames: Either a list of numpy arrays or a single 4D numpy array (N,H,W,C)
        output_path: Path to save the video file
        fps: Frames per second (default 30)
        resize_factor: Factor to resize frames (1.0 means original size)
    """
    if isinstance(frames, np.ndarray):
        if len(frames.shape) != 4:
            raise ValueError("If passing numpy array, must be 4D (N,H,W,C)")
        N, H, W, C = frames.shape
    else:
        if not frames:
            raise ValueError("Empty frames list")
        H, W = frames[0].shape[:2]
        C = frames[0].shape[2] if len(frames[0].shape) == 3 else 1
        N = len(frames)

    # Calculate new dimensions if resizing
    if resize_factor != 1.0:
        new_W = int(W * resize_factor)
        new_H = int(H * resize_factor)
        # Make sure dimensions are even (required by some codecs)
        new_W = new_W + (new_W % 2)
        new_H = new_H + (new_H % 2)
    else:
        new_W, new_H = W, H

    # Use x264 codec with good compression settings
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_W, new_H))

    if not out.isOpened():
        # Fallback to XVID codec if mp4v fails
        out.release()
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_path = output_path.replace(".mp4", ".avi")
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_W, new_H))
        if not out.isOpened():
            raise RuntimeError("Failed to open video writer with both mp4v and XVID codecs")

    try:
        # Write frames with progress bar
        iterator = range(N) if isinstance(frames, np.ndarray) else frames
        for frame in tqdm(iterator, desc="Saving video frames"):
            if isinstance(frames, np.ndarray):
                frame = frames[frame]  # Get frame from numpy array if using range iterator

            # Convert to uint8 if needed
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            # Ensure BGR format for OpenCV
            if C == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Resize if needed
            if resize_factor != 1.0:
                frame = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_AREA)

            out.write(frame)
    finally:
        out.release()

    # After saving, use ffmpeg to compress the video further if available
    try:
        import subprocess

        compressed_path = output_path.replace(".mp4", "_compressed.mp4").replace(".avi", "_compressed.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                output_path,
                "-c:v",
                "libx264",
                "-preset",
                "slow",  # slower preset = better compression
                "-crf",
                "28",  # Constant Rate Factor: 18-28 is good, higher = more compression
                "-y",  # Overwrite output file if it exists
                compressed_path,
            ],
            check=True,
            capture_output=True,
        )

        # Replace original with compressed version if compression succeeded
        os.replace(compressed_path, output_path)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.warning(f"Could not compress video with ffmpeg: {e}")
        logger.warning("Using original video file")


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


class Runner:
    def __init__(
        self,
        embodiment: str,
        policy: ort.InferenceSession,
        config: Dict,
        render: bool = True,
        terrain: bool = False,
        in_the_air: bool = False,
    ) -> None:
        self.policy = policy
        self.render = render
        self.frames = []
        self.framerate = 30
        self.in_the_air = in_the_air
        self.starting_mujoco_pose = {
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

        mujoco_model_path, _ = asyncio.run(get_model_path_and_metadata(embodiment))

        logger.info("Using robot file: %s", os.path.abspath(mujoco_model_path))

        num_actions = 20  # Fixed number of actions for kbot

        # Set up joint mappings first so we have access to joint names
        self.model = load_mjmodel(mujoco_model_path, "smooth")
        self.data = mujoco.MjData(self.model)
        self._setup_joint_mappings(config)

        # Get actuator configs
        kbot_actuators = config["scene"]["robot"]["actuators"]
        kbot_04_cfg = kbot_actuators["kbot_04"]
        kbot_03_cfg = kbot_actuators["kbot_03"]
        kbot_02_cfg = kbot_actuators["kbot_02"]

        # Initialize arrays for actuator parameters
        robot_effort = np.zeros(num_actions)
        robot_stiffness = np.zeros(num_actions)
        robot_damping = np.zeros(num_actions)
        friction_static = np.zeros(num_actions)
        friction_dynamic = np.zeros(num_actions)
        activation_vel = np.zeros(num_actions)

        def get_actuator_cfg_for_joint(name: str) -> dict:
            """Return actuator config block based on joint name suffix."""
            if name.endswith("_04"):
                return kbot_04_cfg
            elif name.endswith("_03"):
                return kbot_03_cfg
            elif name.endswith("_02"):
                return kbot_02_cfg
            else:
                raise ValueError(f"Joint name {name} does not match _02, _03, or _04 suffix")

        # Get the joint names in MuJoCo order
        mujoco_joint_names = []
        if self.in_the_air:
            for ii in range(0, len(self.data.ctrl)):
                mujoco_joint_names.append(self.data.joint(ii).name)
        else:
            for ii in range(1, len(self.data.ctrl) + 1):
                mujoco_joint_names.append(self.data.joint(ii).name)

        # Fill arrays with correct parameters for each joint
        for i, joint_name in enumerate(mujoco_joint_names):
            cfg = get_actuator_cfg_for_joint(joint_name)
            # Add small epsilon to friction values to avoid numerical issues
            FRICTION_EPS = 1e-6

            robot_effort[i] = cfg["effort_limit"]
            robot_stiffness[i] = cfg["stiffness"][".*"]
            robot_damping[i] = cfg["damping"][".*"]
            friction_static[i] = cfg["friction_static"] + FRICTION_EPS
            friction_dynamic[i] = cfg["friction_dynamic"] + FRICTION_EPS
            activation_vel[i] = cfg["activation_vel"]

            # Print parameters for this joint
            logger.debug("\nJoint %s parameters:", joint_name)
            logger.debug("  Effort limit: %f", robot_effort[i])
            logger.debug("  Stiffness: %f", robot_stiffness[i])
            logger.debug("  Damping: %f", robot_damping[i])
            logger.debug("  Static friction: %f", friction_static[i])
            logger.debug("  Dynamic friction: %f", friction_dynamic[i])
            logger.debug("  Activation velocity: %f", activation_vel[i])

        if "joint_pos" in config["actions"]:
            action_scale = config["actions"]["joint_pos"]["scale"]
        elif "relative_joint_pos" in config["actions"]:
            action_scale = config["actions"]["relative_joint_pos"]["scale"]
        else:
            raise ValueError("No action scaling found in config")

        self.model_info = {
            "sim_dt": config["sim"]["dt"],
            "sim_decimation": config["decimation"],
            "tau_factor": [1] * num_actions,
            "num_actions": num_actions,
            "robot_effort": robot_effort,
            "robot_stiffness": robot_stiffness,
            "robot_damping": robot_damping,
            "action_scale": action_scale,
            "friction_static": friction_static,
            "friction_dynamic": friction_dynamic,
            "activation_vel": activation_vel,
            "config": config,
        }
        logger.info("Action scale: %f", self.model_info["action_scale"])
        self.model.opt.timestep = self.model_info["sim_dt"]

        # Set up control parameters
        self.tau_limit = np.array(self.model_info["robot_effort"]) * self.model_info["tau_factor"]
        self.kps = np.array(self.model_info["robot_stiffness"])
        self.kds = np.array(self.model_info["robot_damping"])

        # Initialize default position
        try:
            # First, let's print the actual MuJoCo joint order for verification
            logger.info("MuJoCo joint order:")
            for i, name in enumerate(mujoco_joint_names):
                logger.info("%d: %s", i, name)

            # Define initial pose in MuJoCo joint order
            # Note: This follows the order of mujoco_joint_names, not isaac_joint_names
            initial_pose = np.zeros(self.model_info["num_actions"])

            # Set the angles in the correct order
            for i, joint_name in enumerate(mujoco_joint_names):
                if joint_name in self.starting_mujoco_pose:
                    initial_pose[i] = self.starting_mujoco_pose[joint_name]
                    logger.info(
                        "Setting %s to %f rad (%f deg)",
                        joint_name,
                        self.starting_mujoco_pose[joint_name],
                        np.degrees(self.starting_mujoco_pose[joint_name]),
                    )

            # Set the initial pose
            self.data.qpos[-self.model_info["num_actions"] :] = initial_pose
            self.default = initial_pose.copy()
            logging.info("Default position: %s", self.default)

        except:
            logger.warning("No default position found, using zero initialization")
            self.default = np.zeros(self.model_info["num_actions"])

        # Initialize simulation state
        self.data.qvel = np.zeros_like(self.data.qvel)
        self.data.qacc = np.zeros_like(self.data.qacc)

        # Initialize viewer
        if self.render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        else:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, "offscreen")

        # Initialize control variables
        self.target_q = np.zeros((self.model_info["num_actions"]), dtype=np.double)
        self.last_action = np.zeros((self.model_info["num_actions"]), dtype=np.double)
        self.count_lowlevel = 0
        logger.debug("Model info: %s", self.model_info)

        # breakpoint()

    def _setup_joint_mappings(self, config: Dict) -> None:
        """Set up mappings between MuJoCo and Isaac joint names."""
        mujoco_joint_names = []
        mujoco.mj_step(self.model, self.data)

        if self.in_the_air:
            for ii in range(0, len(self.data.ctrl)):
                mujoco_joint_names.append(self.data.joint(ii).name)
        else:
            for ii in range(1, len(self.data.ctrl) + 1):
                mujoco_joint_names.append(self.data.joint(ii).name)

        # TODO - load this at runtime
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

        print(mujoco_joint_names)

        for ii in range(len(mujoco_joint_names)):
            logging.info("%s -> %s", mujoco_joint_names[ii], isaac_joint_names[ii])

        # Create mappings
        self.mujoco_to_isaac_mapping = {
            mujoco_joint_names[i]: isaac_joint_names[i] for i in range(len(mujoco_joint_names))
        }
        self.isaac_to_mujoco_mapping = {
            isaac_joint_names[i]: mujoco_joint_names[i] for i in range(len(isaac_joint_names))
        }

        # breakpoint()
        # TODO - update the values
        # self.test_mappings()

    def map_isaac_to_mujoco(self, isaac_position: np.ndarray) -> np.ndarray:
        """Maps joint positions from Isaac format to MuJoCo format.

        Args:
            isaac_position (np.array): Joint positions in Isaac order.

        Returns:
            np.array: Joint positions in MuJoCo order.
        """
        mujoco_position = np.zeros(len(self.isaac_to_mujoco_mapping))
        for isaac_index, isaac_name in enumerate(self.isaac_to_mujoco_mapping.keys()):
            mujoco_index = list(self.isaac_to_mujoco_mapping.values()).index(isaac_name)
            mujoco_position[mujoco_index] = isaac_position[isaac_index]
        return mujoco_position

    def map_mujoco_to_isaac(self, mujoco_position):
        """Maps joint positions from MuJoCo format to Isaac format.

        Args:
            mujoco_position (np.array): Joint positions in MuJoCo order.

        Returns:
            np.array: Joint positions in Isaac order.
        """
        # Create an array for Isaac positions based on the mapping
        isaac_position = np.zeros(len(self.mujoco_to_isaac_mapping))
        for mujoco_index, mujoco_name in enumerate(self.mujoco_to_isaac_mapping.keys()):
            isaac_index = list(self.mujoco_to_isaac_mapping.values()).index(mujoco_name)
            isaac_position[isaac_index] = mujoco_position[mujoco_index]
        return isaac_position

    def step(self, x_vel_cmd: float, y_vel_cmd: float, yaw_vel_cmd: float):
        """Execute one step of the simulation.

        Args:
            x_vel_cmd: X velocity command
            y_vel_cmd: Y velocity command
            yaw_vel_cmd: Yaw velocity command
        """
        # Last 20 dof are our "active" joints
        q = self.data.qpos[-self.model_info["num_actions"] :]
        dq = self.data.qvel[-self.model_info["num_actions"] :]

        # Read orientation sensor -> pass to get_gravity_orientation
        # Make sure the sensor name "orientation" is correct in your XML
        # Try both possible sensor names for orientation
        try:
            orientation_quat = self.data.sensor("orientation").data  # shape (4,)

        except Exception:
            try:
                orientation_quat = self.data.sensor("base_link_quat").data  # shape (4,)
            except Exception:
                raise Exception("Could not find orientation sensor - tried names 'orientation' and 'base_link_quat'")
        projected_gravity = get_gravity_orientation(orientation_quat)

        try:
            imu_ang_vel = self.data.sensor("angular-velocity").data
        except Exception:
            try:
                imu_ang_vel = self.data.sensor("imu_gyro").data
            except Exception:
                raise Exception(
                    "Could not find angular velocity sensor - tried names 'angular-velocity' and 'imu_angular_velocity'"
                )

        try:
            imu_lin_acc = self.data.sensor("linear-acceleration").data
        except Exception:
            try:
                imu_lin_acc = self.data.sensor("imu_acc").data
            except Exception:
                raise Exception(
                    "Could not find linear acceleration sensor - tried names 'linear-acceleration' and 'imu_linear_acceleration'"
                )

        # Debug IMU values periodically
        if self.count_lowlevel % 100 == 0:  # Print every 100 steps to avoid spam
            logger.debug("\nIMU Debug:")
            logger.debug("Angular velocity (rad/s): %s", imu_ang_vel)
            logger.debug("Linear acceleration (m/s²): %s", imu_lin_acc)
            logger.debug("Gravity projection: %s", projected_gravity)

        # breakpoint()
        # Build the observation only if it's time to do policy inference
        if self.count_lowlevel % self.model_info["sim_decimation"] == 0:
            # Position offset from default
            cur_pos_obs = q - self.default
            # Joint velocities
            cur_vel_obs = dq

            # 3D velocity commands
            vel_cmd = np.array([x_vel_cmd, y_vel_cmd, yaw_vel_cmd], dtype=np.float32)

            # Map from MuJoCo -> Isaac indexing
            cur_pos_isaac = self.map_mujoco_to_isaac(cur_pos_obs).astype(np.float32)
            cur_vel_isaac = self.map_mujoco_to_isaac(cur_vel_obs).astype(np.float32)
            last_act_isaac = self.last_action.astype(np.float32)

            # Now build the entire 72-D observation:
            #   3 (vel_cmd) + 3 (imu_ang_vel) + 3 (imu_lin_acc) + 3 (proj_grav)
            #   + 20 (joint pos) + 20 (joint vel) + 20 (last_action) = 72
            obs = np.concatenate(
                [
                    vel_cmd,  # 3
                    # imu_lin_acc.astype(np.float32),   # 3
                    # imu_ang_vel.astype(np.float32),   # 3
                    projected_gravity.astype(np.float32),  # 3
                    cur_pos_isaac,  # 20
                    cur_vel_isaac,  # 20
                    last_act_isaac,  # 20
                ],
            )

            logger.debug("obs values:")

            # Command velocities
            logger.debug("  0: cmd_vel_x: %f", obs[0])
            logger.debug("  1: cmd_vel_y: %f", obs[1])
            logger.debug("  2: cmd_vel_yaw: %f", obs[2])

            # Projected gravity
            logger.debug("  3: proj_grav_x: %f", obs[3])
            logger.debug("  4: proj_grav_y: %f", obs[4])
            logger.debug("  5: proj_grav_z: %f", obs[5])

            # Joint positions
            for i in range(20):
                logger.debug("  %d: joint_pos_%d: %f", i + 6, i, obs[i + 6])

            # Joint velocities
            for i in range(20):
                logger.debug("  %d: joint_vel_%d: %f", i + 26, i, obs[i + 26])

            # Last actions
            for i in range(20):
                logger.debug("  %d: last_action_%d: %f", i + 46, i, obs[i + 46])

            # Run the ONNX policy
            input_name = self.policy.get_inputs()[0].name
            curr_actions = self.policy.run(None, {input_name: obs.reshape(1, -1).astype(np.float32)})[0][0]

            # Zero out all actions
            # curr_actions = np.zeros_like(curr_actions)
            # Update last_action
            self.last_action = curr_actions.copy()

            # Scale actions, then map Isaac->MuJoCo indexing

            final_action = None
            if "relative_joint_pos" in self.model_info["config"]["actions"]:

                max_action = self.model_info["config"]["actions"]["relative_joint_pos"]["max_action"]

                # scale 
                curr_actions_scaled = curr_actions * self.model_info["action_scale"]
                # offset 
                curr_actions_scaled_offset = curr_actions_scaled + self.default
                # tanh
                curr_actions_delta = np.tanh(curr_actions_scaled_offset) * max_action
                # relative
                curr_actions_relative = curr_actions_delta + cur_pos_obs
                final_action = curr_actions_relative
            elif "joint_pos" in self.model_info["config"]["actions"]:
                # scale
                curr_actions_scaled = curr_actions * self.model_info["action_scale"]
                # offset
                final_action = curr_actions_scaled
            else:
                raise ValueError("No action scaling found in config")


            self.target_q = self.map_isaac_to_mujoco(final_action)

            # Render if needed
            if self.render:
                self.viewer.render()
            else:
                # Offscreen mode -> record frames for a video
                self.frames.append(self.viewer.read_pixels(camid=0))

        # PD control for the step
        tau = self.kps * (self.default + self.target_q - q) - self.kds * dq

        # add friction logic from Isaac Lab
        # tau -= (
        #     self.model_info["friction_static"] * np.tanh(dq / self.model_info["activation_vel"])
        #     + self.model_info["friction_dynamic"] * dq
        # )

        # Apply torques & step
        self.data.ctrl = tau
        mujoco.mj_step(self.model, self.data)

        self.count_lowlevel += 1

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()

    def save_video(self, filename: str = "episode.mp4"):
        """Save recorded frames as a video file."""
        if not self.render and self.frames:
            save_video_cv2(self.frames, filename, fps=self.framerate, resize_factor=0.75)  # Reduce resolution to 75%


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--embodiment", type=str, default="kbot-v1", help="Embodiment name.")
    parser.add_argument("--sim_duration", type=float, default=5, help="Simulation duration in seconds.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="assets/saved_checkpoints/2025-02-20_00-28-33_model_2600",
        help="Model path.",
    )
    parser.add_argument("--terrain", action="store_true", help="Render the terrain.")
    parser.add_argument("--air", action="store_true", help="Run in the air.")
    parser.add_argument("--render", action="store_true", help="Render the terrain.")
    args = parser.parse_args()

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = -1.9, 0.0, 0.0

    # Get the most recent yaml and onnx files from the checkpoint directory
    yaml_files = [f for f in os.listdir(args.model_path) if f.endswith("env.yaml")]
    policy_files = [f for f in os.listdir(args.model_path) if f.endswith(".onnx")]

    if not yaml_files or not policy_files:
        raise FileNotFoundError(f"Could not find env.yaml and .onnx files in {args.model_path}")

    yaml_file = yaml_files[0]  # Use first found yaml
    policy_file = policy_files[0]  # Use first found onnx

    policy_path = os.path.join(args.model_path, policy_file)
    yaml_path = os.path.join(args.model_path, yaml_file)

    logger.info("Loading policy from: %s", os.path.abspath(policy_path))
    logger.info("Loading config from: %s", os.path.abspath(yaml_path))

    policy = onnx.load(policy_path)
    session = ort.InferenceSession(policy.SerializeToString())

    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    runner = Runner(
        embodiment=args.embodiment,
        policy=session,
        config=config,
        render=args.render,
        terrain=args.terrain,
        in_the_air=args.air,
    )

    for _ in tqdm(range(int(args.sim_duration / config["sim"]["dt"])), desc="Simulating..."):
        runner.step(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)

    # Create mujoco_videos directory in model path if it doesn't exist
    logger.info("Saving video...")
    video_dir = os.path.join(args.model_path, "mujoco_videos")
    os.makedirs(video_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_path = os.path.join(video_dir, f"sim_video_{timestamp}.mp4")
    runner.save_video(video_path)
    logger.info("Saved video to: %s", os.path.abspath(video_path))

    runner.close()
