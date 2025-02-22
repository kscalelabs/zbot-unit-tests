"""Script to read and display current positions of all actuators."""

import argparse
import asyncio
import logging
from dataclasses import dataclass
from typing import List
import time
import os

import colorlogging
from pykos import KOS

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
    Actuator(11, 1, 85.0, 2.0, 100.0, "left_shoulder_pitch_03"),
    Actuator(12, 5, 85.0, 2.0, 100.0, "left_shoulder_roll_03"),
    Actuator(13, 9, 30.0, 1.0, 100.0, "left_shoulder_yaw_02"),
    Actuator(14, 13, 30.0, 1.0, 100.0, "left_elbow_02"),
    Actuator(15, 17, 30.0, 1.0, 100.0, "left_wrist_02"),
    Actuator(21, 3, 85.0, 2.0, 100.0, "right_shoulder_pitch_03"),
    Actuator(22, 7, 85.0, 2.0, 100.0, "right_shoulder_roll_03"),
    Actuator(23, 11, 30.0, 1.0, 100.0, "right_shoulder_yaw_02"),
    Actuator(24, 15, 30.0, 1.0, 100.0, "right_elbow_02"),
    Actuator(25, 19, 30.0, 1.0, 100.0, "right_wrist_02"),
    Actuator(31, 0, 85.0, 3.0, 100.0, "left_hip_pitch_04"),
    Actuator(32, 4, 85.0, 2.0, 100.0, "left_hip_roll_03"),
    Actuator(33, 8, 50.0, 5.0, 100.0, "left_hip_yaw_03"),
    Actuator(34, 12, 60.0, 2.0, 100.0, "left_knee_04"),
    Actuator(35, 16, 80.0, 1.0, 100.0, "left_ankle_02"),
    Actuator(41, 2, 85.0, 3.0, 100.0, "right_hip_pitch_04"),
    Actuator(42, 6, 85.0, 2.0, 100.0, "right_hip_roll_03"),
    Actuator(43, 10, 30.0, 2.0, 100.0, "right_hip_yaw_03"),
    Actuator(44, 14, 60.0, 2.0, 100.0, "right_knee_04"),
    Actuator(45, 18, 80.0, 1.0, 100.0, "right_ankle_02"),
]


async def read_positions(kos: KOS, continuous: bool = False, interval: float = 0.1) -> None:
    """Read and display current positions of all actuators."""
    actuator_ids = [actuator.actuator_id for actuator in ACTUATOR_LIST]

    try:
        while True:
            states = await kos.actuator.get_actuators_state(actuator_ids)

            # Clear screen (works on both Windows and Unix)
            print("\033[2J\033[H")

            print("\nCurrent Actuator Positions")
            print("-" * 65)
            print(f"{'Joint Name':<25} {'ID':>4} {'Position':>10} {'Velocity':>10} {'Torque':>10}")
            print("-" * 65)

            for state in states.states:
                actuator = next(a for a in ACTUATOR_LIST if a.actuator_id == state.actuator_id)
                print(
                    f"{actuator.joint_name:<25} {actuator.actuator_id:>4} "
                    f"{state.position:>10.2f} {state.velocity:>10.2f} {state.torque:>10.2f}"
                )

            if not continuous:
                break

            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring terminated by user")


async def main() -> None:
    """Main function to run the position reader."""
    parser = argparse.ArgumentParser(description="Read actuator positions")
    parser.add_argument("--host", type=str, default="localhost", help="KOS server host")
    parser.add_argument("--port", type=int, default=50051, help="KOS server port")
    parser.add_argument("--real", action="store_true", help="Connect to real robot")
    parser.add_argument("--continuous", action="store_true", help="Continuously monitor positions")
    parser.add_argument("--interval", type=float, default=0.1, help="Update interval for continuous mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    colorlogging.configure(level=log_level)

    # host = "100.89.14.31" if args.real else args.host
    host = "100.69.185.128" if args.real else args.host

    try:
        async with KOS(ip=host, port=args.port) as kos:
            if args.continuous:
                logger.info(f"Starting continuous monitoring (Press Ctrl+C to stop)...")
            await read_positions(kos, args.continuous, args.interval)
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nProgram terminated by user")
