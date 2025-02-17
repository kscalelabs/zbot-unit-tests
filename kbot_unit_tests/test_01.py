"""Interactive example script for playing a trajectory from CSV."""

import argparse
import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass
import os
import csv

import colorlogging
import numpy as np
from pykos import KOS
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


ACTUATOR_LIST: list[Actuator] = [
    Actuator(11, 1, 20.0, 2.0, 20.0, "left_shoulder_pitch"),
    Actuator(12, 5, 20.0, 2.0, 20.0, "left_shoulder_roll"),
    Actuator(13, 9, 20.0, 2.0, 20.0, "left_shoulder_yaw"),
    Actuator(14, 13, 20.0, 2.0, 20.0, "left_elbow"),
    Actuator(15, 17, 20.0, 2.0, 20.0, "left_wrist"),
    Actuator(21, 3, 20.0, 2.0, 20.0, "right_shoulder_pitch"),
    Actuator(22, 7, 20.0, 2.0, 20.0, "right_shoulder_roll"),
    Actuator(23, 11, 20.0, 2.0, 20.0, "right_shoulder_yaw"),
    Actuator(24, 15, 20.0, 2.0, 20.0, "right_elbow"),
    Actuator(25, 19, 20.0, 2.0, 20.0, "right_wrist"),
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


def load_csv_data(filename: str, scale_factor: float = 2.0) -> np.ndarray:
    """Load and scale trajectory data from CSV file."""
    data = []
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip header row
        logger.info(f"CSV Header: {header}")

        # Print first row for debugging
        first_row = next(csv_reader)
        values = [float(x) for x in first_row]
        logger.info(f"First row raw values: {values}")
        action_values = [x * scale_factor for x in values[2:]]  # Skip step and env
        logger.info(f"First row action values after scaling: {action_values}")
        data.append(action_values)

        # Load the rest of the data
        for row in csv_reader:
            values = [float(x) for x in row]
            action_values = [x * scale_factor for x in values[2:]]
            data.append(action_values)

    return np.array(data)


async def configure_robot(kos: KOS) -> None:
    """Configure all actuators with their respective gains."""
    logger.info("Configuring robot actuators...")
    for actuator in ACTUATOR_LIST:
        await kos.actuator.configure_actuator(
            actuator_id=actuator.actuator_id,
            kp=actuator.kp,
            kd=actuator.kd,
            max_torque=actuator.max_torque,
            torque_enabled=True,
        )


async def play_trajectory(kos: KOS, trajectory_data: np.ndarray, dt: float = 1 / 50) -> None:
    """Play the trajectory data on the robot with fixed time step.

    Args:
        kos: KOS instance for robot control
        trajectory_data: Array of trajectory positions
        dt: Time step in seconds (default: 1/50s = 50Hz)
    """
    logger.info(f"Playing trajectory with {len(trajectory_data)} timesteps")
    logger.info(f"Trajectory data shape: {trajectory_data.shape}")
    logger.info(f"Control frequency: {1/dt:.1f}Hz")

    # Create a mapping from actuator IDs to CSV column indices for leg motors
    id_to_column = {
        31: 0,  # left_hip_pitch_04 -> column 0
        32: 1,  # left_hip_roll_03 -> column 1
        33: 2,  # left_hip_yaw_03 -> column 2
        34: 3,  # left_knee_04 -> column 3
        35: 4,  # left_ankle_02 -> column 4
        41: 5,  # right_hip_pitch_04 -> column 5
        42: 6,  # right_hip_roll_03 -> column 6
        43: 7,  # right_hip_yaw_03 -> column 7
        44: 8,  # right_knee_04 -> column 8
        45: 9,  # right_ankle_02 -> column 9
    }

    # Create actuator lookup map
    actuator_map = {actuator.actuator_id: actuator for actuator in ACTUATOR_LIST}

    try:
        start_time = time.time()
        next_time = start_time + dt

        for step_idx, step in enumerate(trajectory_data):
            current_time = time.time()
            commands = []

            # Process leg motors from CSV data
            for motor_id, column_idx in id_to_column.items():
                actuator = actuator_map[motor_id]
                try:
                    position = step[column_idx]
                    # Negate left knee values
                    if motor_id == 34:  # left knee
                        position = position * -1.0
                    commands.append({"actuator_id": motor_id, "position": position})
                except IndexError:
                    logger.error(f"Could not get data for {actuator.joint_name} (motor {motor_id})")

            # Send commands to robot
            await kos.actuator.command_actuators(commands)

            # Sleep precisely until next control step
            if next_time > current_time:
                await asyncio.sleep(next_time - current_time)
            next_time += dt

    except KeyboardInterrupt:
        logger.info("\nStopping trajectory playback")


async def main() -> None:
    """Main function to run the trajectory playback."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--sim-only", action="store_true", help="Run simulation only without connecting to the real robot"
    )
    parser.add_argument("--scale", type=float, default=10.0, help="Scaling factor for trajectory values")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    # Load CSV data
    csv_path = "../assets/recording_01/policy_outputs.csv"
    if not os.path.exists(csv_path):
        logger.error(f"Could not find CSV file at {csv_path}")
        return

    try:
        trajectory_data = load_csv_data(csv_path, args.scale)
        logger.info(f"Actions scaled by factor of {args.scale}")
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return

    # Start simulation
    sim_process = subprocess.Popen(["kos-sim", "kbot-v1", "--no-gravity"])
    time.sleep(2)

    try:
        if args.sim_only:
            logger.info("Running in simulation-only mode...")
            async with KOS(ip=args.host, port=args.port) as sim_kos:
                await sim_kos.sim.reset()
                await configure_robot(sim_kos)

                # Set arm motors to zero position
                arm_commands = [
                    {"actuator_id": actuator.actuator_id, "position": 0.0}
                    for actuator in ACTUATOR_LIST
                    if actuator.actuator_id in range(11, 26)
                ]
                await sim_kos.actuator.command_actuators(arm_commands)

                # Play trajectory
                await play_trajectory(sim_kos, trajectory_data)
        else:
            logger.info("Running on both simulator and real robot...")
            async with KOS(ip=args.host, port=args.port) as sim_kos, KOS(ip="10.33.11.164", port=args.port) as real_kos:
                await sim_kos.sim.reset()
                await configure_robot(sim_kos)
                await configure_robot(real_kos)

                # Set arm motors to zero position on both robots
                arm_commands = [
                    {"actuator_id": actuator.actuator_id, "position": 0.0}
                    for actuator in ACTUATOR_LIST
                    if actuator.actuator_id in range(11, 26)
                ]
                await asyncio.gather(
                    sim_kos.actuator.command_actuators(arm_commands),
                    real_kos.actuator.command_actuators(arm_commands),
                )

                # Play trajectory on both robots
                await asyncio.gather(
                    play_trajectory(sim_kos, trajectory_data),
                    play_trajectory(real_kos, trajectory_data),
                )

    except Exception as e:
        logger.error(f"Error: {e}")

    finally:
        sim_process.terminate()
        sim_process.wait()


if __name__ == "__main__":
    asyncio.run(main())
