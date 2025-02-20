"""Walking policy deployment script for simulated K-Bot."""

import argparse
import asyncio
import logging
import time
from typing import Dict, List, Union

import colorlogging
import numpy as np
from kinfer.inference.python import ONNXModel
from pykos import KOS
from scipy.spatial.transform import Rotation as R

# Joint configuration from real robot
JOINT_NAME_LIST = [
    "left_hip_pitch_04",  # Left leg, top to bottom
    "left_hip_roll_03",
    "left_hip_yaw_03",
    "left_knee_04",
    "left_ankle_02",
    "right_hip_pitch_04",  # Right leg, top to bottom
    "right_hip_roll_03",
    "right_hip_yaw_03",
    "right_knee_04",
    "right_ankle_02",
]

JOINT_NAME_TO_ID = {
    # Left leg
    "left_hip_pitch_04": 31,
    "left_hip_roll_03": 32,
    "left_hip_yaw_03": 33,
    "left_knee_04": 34,
    "left_ankle_02": 35,
    # Right leg
    "right_hip_pitch_04": 41,
    "right_hip_roll_03": 42,
    "right_hip_yaw_03": 43,
    "right_knee_04": 44,
    "right_ankle_02": 45,
}

JOINT_SIGNS = {
    # Left leg
    "left_hip_pitch_04": 1,
    "left_hip_roll_03": 1,
    "left_hip_yaw_03": 1,
    "left_knee_04": 1,
    "left_ankle_02": 1,
    # Right leg
    "right_hip_pitch_04": 1,
    "right_hip_roll_03": 1,
    "right_hip_yaw_03": 1,
    "right_knee_04": 1,
    "right_ankle_02": -1,
}

MOTOR_TYPE_TO_METADATA_INDEX = {
    "04": 0,
    "03": 1,
    "02": 2,
}

logger = logging.getLogger(__name__)


class RobotState:
    """Tracks robot state and handles offsets."""

    def __init__(self, joint_names: List[str], joint_signs: Dict[str, float]):
        self.joint_offsets = {name: 0.0 for name in joint_names}
        self.joint_signs = joint_signs
        self.orn_offset = None

    async def offset_in_place(self, kos: KOS, joint_names: List[str]) -> None:
        """Capture current position as zero offset."""
        # Get current joint positions (in degrees)
        states = await kos.actuator.get_actuators_state([JOINT_NAME_TO_ID[name] for name in joint_names])
        current_positions = {name: states.states[i].position for i, name in enumerate(joint_names)}

        # Store negative of current positions as offsets (in degrees)
        self.joint_offsets = {name: 0.0 for name, _ in current_positions.items()}

        # Store IMU offset
        imu_data = await kos.imu.get_euler_angles()
        initial_quat = R.from_euler("xyz", [imu_data.roll, imu_data.pitch, imu_data.yaw], degrees=True).as_quat()
        self.orn_offset = R.from_quat(initial_quat).inv()

    async def get_obs(self, kos: KOS) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get robot state with offset compensation."""
        # Batch state requests
        states, euler_data, imu_sensor_data = await asyncio.gather(
            kos.actuator.get_actuators_state([JOINT_NAME_TO_ID[name] for name in JOINT_NAME_LIST]),
            kos.imu.get_euler_angles(),
            kos.imu.get_imu_values(),
        )

        # Apply offsets and signs to positions and convert to radians
        q = np.array(
            [
                np.deg2rad((states.states[i].position + self.joint_offsets[name]) * self.joint_signs[name])
                for i, name in enumerate(JOINT_NAME_LIST)
            ],
            dtype=np.float32,
        )

        # Apply signs to velocities and convert to radians
        dq = np.array(
            [np.deg2rad(states.states[i].velocity * self.joint_signs[name]) for i, name in enumerate(JOINT_NAME_LIST)],
            dtype=np.float32,
        )

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

        return q, dq, quat, gvec, omega

    def apply_command(self, position: float, joint_name: str) -> float:
        """Apply sign first, then offset to outgoing command. Convert from radians to degrees."""
        position_deg = np.rad2deg(position)
        return position_deg * self.joint_signs[joint_name] - self.joint_offsets[joint_name]


async def run_robot(
    kos: KOS,
    policy: ONNXModel,
    model_info: Dict[str, Union[float, List[float], str]],
    x_vel: float = 0.2,
    y_vel: float = 0.0,
    yaw_vel: float = 0.0,
) -> None:
    """Run the walking policy on the simulated robot."""
    # Initialize robot state handler
    robot_state = RobotState(JOINT_NAME_LIST, JOINT_SIGNS)

    # Configure motors
    logger.info("Configuring motors...")
    leg_ids = [JOINT_NAME_TO_ID[name] for name in JOINT_NAME_LIST]

    # Capture current position as zero
    logger.info("Capturing current position as zero...")
    await robot_state.offset_in_place(kos, JOINT_NAME_LIST)

    efforts = model_info["robot_effort"]
    tau_limit = (
        np.array([efforts[MOTOR_TYPE_TO_METADATA_INDEX[name[-2:]]] for name in JOINT_NAME_LIST])
        * model_info["tau_factor"]
    )
    p_gains = model_info["robot_stiffness"]
    kps = np.array([p_gains[MOTOR_TYPE_TO_METADATA_INDEX[name[-2:]]] for name in JOINT_NAME_LIST])
    d_gains = model_info["robot_damping"]
    kds = np.array([d_gains[MOTOR_TYPE_TO_METADATA_INDEX[name[-2:]]] for name in JOINT_NAME_LIST])

    # Configure gains for each joint
    for i, joint_name in enumerate(JOINT_NAME_LIST):
        joint_id = JOINT_NAME_TO_ID[joint_name]
        logger.debug(
            f"Configuring joint {joint_name} with ID {joint_id} for kp={kps[i]}, kd={kds[i]}, max_torque={tau_limit[i]}"
        )
        await kos.actuator.configure_actuator(
            actuator_id=joint_id,
            kp=float(kps[i]),
            kd=float(kds[i]),
            max_torque=float(tau_limit[i]),
            torque_enabled=True,
        )

    # Initialize policy state
    default = np.array(model_info["default_standing"])
    target_q = np.zeros(model_info["num_actions"], dtype=np.float32)
    prev_actions = np.zeros(model_info["num_actions"], dtype=np.float32)
    hist_obs = np.zeros(model_info["num_observations"], dtype=np.float32)
    count_policy = 0

    logger.info("Going to zero position...")
    await kos.actuator.command_actuators([{"actuator_id": joint_id, "position": 0.0} for joint_id in leg_ids])

    for i in range(3, -1, -1):
        logger.info(f"Starting in {i} seconds...")
        await asyncio.sleep(1)

    try:
        logger.info("Starting control loop...")
        while True:
            process_start = time.time()

            try:
                obs_start = time.time()
                # Get robot state with offset compensation
                q, dq, quat, gvec, omega = await robot_state.get_obs(kos)
                obs_time = time.time() - obs_start

                logger.debug("Obs time: %s", obs_time)

                # Prepare policy inputs and run policy
                input_data = {
                    "x_vel.1": np.array([x_vel], dtype=np.float32),
                    "y_vel.1": np.array([y_vel], dtype=np.float32),
                    "rot.1": np.array([yaw_vel], dtype=np.float32),
                    "t.1": np.array([count_policy * model_info["policy_dt"]], dtype=np.float32),
                    "dof_pos.1": (q - default).astype(np.float32),
                    "dof_vel.1": dq.astype(np.float32),
                    "prev_actions.1": prev_actions.astype(np.float32),
                    "projected_gravity.1": gvec.astype(np.float32),
                    "imu_ang_vel.1": omega.astype(np.float32),
                    "buffer.1": hist_obs.astype(np.float32),
                }

                logger.debug("q: %s", q)
                logger.debug("dq: %s", dq)
                logger.debug("quat: %s", quat)
                logger.debug("gvec: %s", gvec)
                logger.debug("omega: %s", omega)

                # Run policy
                policy_output = policy(input_data)
                target_q = policy_output["actions_scaled"]
                prev_actions = policy_output["actions"]
                hist_obs = policy_output["x.3"]

                # Apply commands with offset compensation
                commands = []
                for i, joint_name in enumerate(JOINT_NAME_LIST):
                    joint_id = JOINT_NAME_TO_ID[joint_name]
                    position = robot_state.apply_command(float(target_q[i] + default[i]), joint_name)
                    commands.append({"actuator_id": joint_id, "position": position})

                logger.debug("Sending commands to actuators")
                await kos.actuator.command_actuators(commands)

                process_time = time.time() - process_start
                sleep = model_info["policy_dt"] - process_time
                if sleep > 0:
                    logger.debug(f"Sleeping for {sleep:.4f} seconds")
                    await asyncio.sleep(sleep)
                else:
                    logger.debug(f"Overstepped policy time by {sleep:.4f} seconds")

                count_policy += 1

            except asyncio.CancelledError:
                raise

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("\nStopping walking...")
    finally:
        # Disable torque on exit
        logger.info("Disabling torque on all joints...")
        for joint_id in leg_ids:
            await kos.actuator.configure_actuator(actuator_id=joint_id, torque_enabled=False)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated robot walking deployment script.")
    parser.add_argument("--load_model", type=str, required=True, help="Path to policy model")
    parser.add_argument("--host", type=str, default="localhost", help="Simulation host")
    parser.add_argument("--port", type=int, default=50051, help="Simulation port")
    parser.add_argument("--x_vel", type=float, default=0.2, help="X velocity")
    parser.add_argument("--y_vel", type=float, default=0.0, help="Y velocity")
    parser.add_argument("--yaw_vel", type=float, default=0.0, help="Yaw velocity")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging
    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    # Initialize KOS and policy
    logger.info("Starting PPO walking client...")
    async with KOS(ip=args.host, port=args.port) as kos:
        policy = ONNXModel(args.load_model)

        # Get model info from policy metadata
        metadata = policy.get_metadata()
        model_info = {
            "num_actions": metadata["num_actions"],
            "num_observations": metadata["num_observations"],
            "robot_effort": metadata["robot_effort"],
            "robot_stiffness": metadata["robot_stiffness"],
            "robot_damping": metadata["robot_damping"],
            "tau_factor": metadata["tau_factor"],
            "policy_dt": metadata["sim_dt"] * metadata["sim_decimation"],
            "default_standing": metadata["default_standing"],
        }

        logger.info(f"Model Info: {model_info}")

        # Reset simulation
        await kos.sim.reset()

        # Run robot control
        await run_robot(
            kos=kos,
            policy=policy,
            model_info=model_info,
            x_vel=args.x_vel,
            y_vel=args.y_vel,
            yaw_vel=args.yaw_vel,
        )


if __name__ == "__main__":
    asyncio.run(main())
