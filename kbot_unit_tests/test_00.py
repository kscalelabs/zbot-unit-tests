"""
Simple Bilateral Movement Test with LEFT side inverted
Tests that bilateral joints move together synchronously in the same directions.
Includes actuator configuration with standardized max torque.
"""

import asyncio
import logging
import colorlogging
import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
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


async def smooth_move_joints(
    kos: KOS,
    actuator_ids: List[int],
    start_positions: List[float],
    end_positions: List[float],
    duration: float = 0.8,
    steps: int = 20,
) -> None:
    """
    Smoothly move joints from start positions to end positions using
    a sinusoidal velocity profile for smooth acceleration/deceleration.

    Args:
        kos: KOS instance
        actuator_ids: List of actuator IDs to move
        start_positions: Starting positions for each actuator
        end_positions: Target positions for each actuator
        duration: Total movement duration in seconds
        steps: Number of intermediate steps
    """
    if len(actuator_ids) != len(start_positions) or len(actuator_ids) != len(end_positions):
        raise ValueError("Number of actuator IDs must match number of positions")

    # Calculate time step
    time_step = duration / steps

    # For each time step
    for step in range(steps + 1):
        # Calculate progress (0.0 to 1.0)
        t = step / steps

        # Use sinusoidal easing for smooth acceleration/deceleration
        # This creates an S-curve velocity profile
        smoothed_t = (1 - math.cos(t * math.pi)) / 2

        # Calculate interpolated position for each actuator
        commands = []
        for i, actuator_id in enumerate(actuator_ids):
            start_pos = start_positions[i]
            end_pos = end_positions[i]
            current_pos = start_pos + (end_pos - start_pos) * smoothed_t
            commands.append({"actuator_id": actuator_id, "position": current_pos})

        # Send commands to actuators
        await kos.actuator.command_actuators(commands)

        # Wait for the next step
        await asyncio.sleep(time_step)


async def configure_actuators(kos):
    """
    Configure all actuators with balanced gains for responsive yet smooth movement.
    Slightly lower kp values maintain control while reducing stiffness.
    Slightly higher kd values provide just enough damping without excessive slowdown.
    """
    logger.info("Configuring actuators for responsive, smooth movement...")
    for actuator in ACTUATOR_LIST:
        try:
            # Reduce kp by just 10% for better responsiveness while still softer
            adjusted_kp = actuator.kp * 0.9
            # Increase kd by 20% - enough damping without excessive slowdown
            adjusted_kd = actuator.kd * 1.2

            logger.info(
                f"Configuring {actuator.joint_name} (ID: {actuator.actuator_id}) for smooth movement:\n"
                f"    kp: {adjusted_kp} (original: {actuator.kp})\n"
                f"    kd: {adjusted_kd} (original: {actuator.kd})\n"
                f"    max_torque: {actuator.max_torque}"
            )
            await kos.actuator.configure_actuator(
                actuator_id=actuator.actuator_id,
                kp=adjusted_kp,
                kd=adjusted_kd,
                max_torque=actuator.max_torque,
                torque_enabled=True,
            )
        except Exception as e:
            logger.error(f"Failed to configure {actuator.joint_name}: {e}")
    logger.info("All actuators configured for smooth movement.")


async def test_bilateral_movements(kos):
    """
    Tests that bilateral joints move together synchronously.
    Simply moves joint pairs up, down, and to zero together.
    LEFT side joints are inverted to ensure physical movement is synchronized.
    Uses interpolation and gradual acceleration to create smooth movements.
    """
    logger.info("Starting bilateral movement test (with LEFT side inverted)")

    # Define joint pairs we want to test with correct UP and DOWN movements
    joint_pairs = [
        {
            "name": "ankles",
            "left_id": 35,
            "right_id": 45,
            # UP is dorsiflexion (foot moves upward toward shin)
            "down_pos": -20.0,  # NEGATIVE is DOWN for ankles (pointed toes)
            "up_pos": 20.0,  # POSITIVE is UP for ankles (toes toward shin)
            "invert_left": True,  # Set to True if left side needs inverting
        },
        {
            "name": "knees",
            "left_id": 34,
            "right_id": 44,
            # UP is flexion (bend knee toward sitting position)
            "up_pos": -45.0,  # NEGATIVE is BENT for knees
            "down_pos": 0.0,  # ZERO is DOWN for knees (straight leg)
            "invert_left": True,  # Left knee needs inverting
        },
        {
            "name": "hip_yaw",
            "left_id": 33,
            "right_id": 43,
            "up_pos": -45.0,  # this means legs turned OUT
            "down_pos": 45.0,  # this means legs turned IN
            "invert_left": True,
        },
        {
            "name": "hip_pitch",
            "left_id": 31,
            "right_id": 41,
            # UP is flexion (like lifting leg forward/up)
            "up_pos": -20.0,  # POSITIVE is UP for hip pitch
            "down_pos": 20.0,  # NEGATIVE is DOWN for hip pitch
            "invert_left": True,  # Left hip pitch needs inverting
        },
        {
            "name": "hip_roll",
            "left_id": 32,
            "right_id": 42,
            # UP is abduction (leg moves away from midline)
            "up_pos": 20.0,  # LEFT moves OUT
            "down_pos": -15.0,  # LEFT moves IN
            "invert_left": True,  # Roll joints keep expected mirroring
        },
        {
            "name": "shoulder_pitch",
            "left_id": 11,
            "right_id": 21,
            # UP is flexion (raising arm forward/up)
            "up_pos": -45.0,  # POSITIVE is UP for shoulder pitch
            "down_pos": 30.0,  # NEGATIVE is DOWN for shoulder pitch
            "invert_left": True,  # Left shoulder needs inverting
        },
        {
            "name": "shoulder_roll",
            "left_id": 12,
            "right_id": 22,
            # UP is abduction (arm moves away from midline)
            "up_pos": 45.0,  #  moves OUT with Positive
            "down_pos": -45.0,  # moves IN with negative
            "invert_left": True,  # Roll joints keep expected mirroring
        },
        {
            "name": "shoulder_yaw",
            "left_id": 13,
            "right_id": 23,
            "up_pos": -45.0,
            "down_pos": 45.0,
            "invert_left": True,
        },
        {
            "name": "elbows",
            "left_id": 14,
            "right_id": 24,
            # UP is flexion (bending elbow so hand moves toward shoulder)
            "up_pos": 90.0,  # POSITIVE is UP for elbows (bend)
            "down_pos": 0.0,  # ZERO is DOWN for elbows (straight)
            "invert_left": True,  # Left elbow needs inverting
        },
        {
            "name": "wrist_pitch",
            "left_id": 15,
            "right_id": 25,
            "up_pos": -45.0,  # up is OUT
            "down_pos": 45.0,  # down is IN
            "invert_left": True,
        },
    ]

    # First reset all joints with smooth motion
    logger.info("RESETTING all joints to zero position with smooth interpolation...")
    all_actuator_ids = []
    all_target_positions = []

    # Build lists ensuring they stay in sync
    for pair in joint_pairs:
        all_actuator_ids.extend([pair["left_id"], pair["right_id"]])
        all_target_positions.extend([0.0, 0.0])  # Zero position for both left and right

    # Get current positions of all joints
    states = await kos.actuator.get_actuators_state(all_actuator_ids)
    current_positions = []
    for state in states.states:
        current_positions.append(state.position)

    # Verify arrays are same length before moving
    if len(all_actuator_ids) == len(current_positions) == len(all_target_positions):
        # Smoothly move all joints to zero
        await smooth_move_joints(
            kos,
            all_actuator_ids,
            current_positions,
            all_target_positions,
            duration=1.2,  # Faster but still controlled initial reset
        )
    else:
        logger.error(
            f"Array length mismatch: ids={len(all_actuator_ids)}, current={len(current_positions)}, target={len(all_target_positions)}"
        )

    await asyncio.sleep(0.5)

    # Test each joint pair
    for pair in joint_pairs:
        left_multiplier = -1 if pair["invert_left"] else 1

        # Test UP position with smooth interpolation
        logger.info(f"MOVING {pair['name']} UP...")
        left_pos = pair["up_pos"] * left_multiplier
        right_pos = pair["up_pos"]

        # Get current positions
        states = await kos.actuator.get_actuators_state([pair["left_id"], pair["right_id"]])
        left_start = states.states[0].position
        right_start = states.states[1].position

        # Perform smooth interpolated movement
        await smooth_move_joints(
            kos, [pair["left_id"], pair["right_id"]], [left_start, right_start], [left_pos, right_pos], duration=2.0
        )

        # Wait a bit for stabilization
        await asyncio.sleep(0.5)
        states = await kos.actuator.get_actuators_state([pair["left_id"], pair["right_id"]])
        left_actual = states.states[0].position
        right_actual = states.states[1].position

        left_ok = abs(left_actual - left_pos) < 8.0
        right_ok = abs(right_actual - right_pos) < 8.0

        logger.info(f"  Left: {left_actual:.1f}° (expected {left_pos:.1f}°) - {'✓' if left_ok else '✗'}")
        logger.info(f"  Right: {right_actual:.1f}° (expected {right_pos:.1f}°) - {'✓' if right_ok else '✗'}")
        logger.info(f"  Synchronized: {'YES' if left_ok and right_ok else 'NO'}")

        # Test ZERO position with smooth interpolation
        logger.info(f"MOVING {pair['name']} to ZERO...")

        # Get current positions
        states = await kos.actuator.get_actuators_state([pair["left_id"], pair["right_id"]])
        left_start = states.states[0].position
        right_start = states.states[1].position

        # Perform smooth interpolated movement
        await smooth_move_joints(
            kos, [pair["left_id"], pair["right_id"]], [left_start, right_start], [0.0, 0.0], duration=0.8
        )

        # Shorter stabilization time
        await asyncio.sleep(0.2)
        states = await kos.actuator.get_actuators_state([pair["left_id"], pair["right_id"]])
        left_actual = states.states[0].position
        right_actual = states.states[1].position

        left_ok = abs(left_actual) < 5.0
        right_ok = abs(right_actual) < 5.0

        logger.info(f"  Left: {left_actual:.1f}° (expected 0.0°) - {'✓' if left_ok else '✗'}")
        logger.info(f"  Right: {right_actual:.1f}° (expected 0.0°) - {'✓' if right_ok else '✗'}")
        logger.info(f"  Synchronized: {'YES' if left_ok and right_ok else 'NO'}")

        # Test DOWN position with smooth interpolation
        logger.info(f"MOVING {pair['name']} DOWN...")
        left_pos = pair["down_pos"] * left_multiplier
        right_pos = pair["down_pos"]

        # Get current positions
        states = await kos.actuator.get_actuators_state([pair["left_id"], pair["right_id"]])
        left_start = states.states[0].position
        right_start = states.states[1].position

        # Perform smooth interpolated movement
        await smooth_move_joints(
            kos, [pair["left_id"], pair["right_id"]], [left_start, right_start], [left_pos, right_pos], duration=2.0
        )

        # Wait a bit for stabilization
        await asyncio.sleep(0.5)
        states = await kos.actuator.get_actuators_state([pair["left_id"], pair["right_id"]])
        left_actual = states.states[0].position
        right_actual = states.states[1].position

        left_ok = abs(left_actual - left_pos) < 8.0
        right_ok = abs(right_actual - right_pos) < 8.0

        logger.info(f"  Left: {left_actual:.1f}° (expected {left_pos:.1f}°) - {'✓' if left_ok else '✗'}")
        logger.info(f"  Right: {right_actual:.1f}° (expected {right_pos:.1f}°) - {'✓' if right_ok else '✗'}")
        logger.info(f"  Synchronized: {'YES' if left_ok and right_ok else 'NO'}")

        # Reset before testing next pair with smooth interpolation
        logger.info(f"RESETTING {pair['name']} to zero position...")

        # Get current positions
        states = await kos.actuator.get_actuators_state([pair["left_id"], pair["right_id"]])
        left_start = states.states[0].position
        right_start = states.states[1].position

        # Perform smooth interpolated movement
        await smooth_move_joints(
            kos, [pair["left_id"], pair["right_id"]], [left_start, right_start], [0.0, 0.0], duration=2.0
        )

        # Wait a bit for stabilization
        await asyncio.sleep(0.5)

    logger.info("Bilateral movement test complete!")


async def main():
    parser = argparse.ArgumentParser(description="Simple Bilateral Movement Test (LEFT INVERTED)")
    parser.add_argument("--host", type=str, default="100.89.14.31", help="KOS server host")
    parser.add_argument("--port", type=int, default=50051, help="KOS server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    colorlogging.configure(level=log_level)

    logger.info(f"Connecting to KOS at {args.host}:{args.port}")
    try:
        async with KOS(ip=args.host, port=args.port) as kos:
            # First configure all actuators
            await configure_actuators(kos)
            # Then run the bilateral movement test
            await test_bilateral_movements(kos)
            logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
