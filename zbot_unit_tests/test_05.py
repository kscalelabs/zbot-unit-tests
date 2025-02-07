"""Tests running ZMP-based walking on the real robot."""

import asyncio
import logging
import math
import time
from typing import List, Tuple

import colorlogging
import pykos

logger = logging.getLogger(__name__)

# Actuator IDs from test_01.py
LEFT_LEG_ACTUATORS = [31, 32, 33, 34, 35]
RIGHT_LEG_ACTUATORS = [41, 42, 43, 44, 45]

# ZMP walking parameters
STEP_DURATION = 0.8  # seconds per step
STEP_HEIGHT = 0.04  # meters
STEP_LENGTH = 0.08  # meters
COM_HEIGHT = 0.35  # meters (height of center of mass)
FOOT_SEPARATION = 0.12  # meters (distance between feet)


async def main() -> None:
    colorlogging.configure()
    logger.warning("Starting test-05")
    try:
        async with pykos.KOS("192.168.42.1") as kos:
            await run_zmp_walking(kos)
    except Exception:
        logger.exception("Make sure that the Z-Bot is connected over USB and the IP address is accessible.")
        raise


def generate_foot_trajectory(t: float, step_phase: float, is_swing_foot: bool) -> Tuple[float, float, float]:
    """Generates foot trajectory for a single step.

    Args:
        t: Current time in the step cycle (0 to STEP_DURATION)
        step_phase: Phase of the walking cycle (0 to 1)
        is_swing_foot: Whether this is the swing foot

    Returns:
        Tuple of (x, y, z) positions for the foot
    """
    if not is_swing_foot:
        # Stance foot stays planted
        return 0.0, (FOOT_SEPARATION / 2), 0.0

    # Swing foot follows a semi-circular trajectory
    phase = t / STEP_DURATION
    x = STEP_LENGTH * (phase - 0.5)
    z = STEP_HEIGHT * math.sin(phase * math.pi)
    y = FOOT_SEPARATION / 2

    return x, y, z


def compute_com_trajectory(t: float, step_phase: float) -> Tuple[float, float, float]:
    """Computes the desired center of mass position.

    Args:
        t: Current time in the step cycle
        step_phase: Phase of the walking cycle

    Returns:
        Tuple of (x, y, z) positions for the center of mass
    """
    # Simple pendulum model for lateral motion
    y = (FOOT_SEPARATION / 2) * math.cos(2 * math.pi * step_phase)

    # Linear forward motion
    x = (STEP_LENGTH / 2) * (t / STEP_DURATION)

    # Constant height
    z = COM_HEIGHT

    return x, y, z


async def configure_leg_actuators(kos: pykos.KOS, actuator_ids: List[int]) -> None:
    """Configures actuators for a leg."""
    for actuator_id in actuator_ids:
        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=64.0,  # Higher gains for better position tracking
            kd=4.0,
            torque_enabled=True,
        )


async def run_zmp_walking(kos: pykos.KOS) -> None:
    """Implements ZMP-based walking on the Z-Bot."""
    logger.info("Configuring actuators...")
    await configure_leg_actuators(kos, LEFT_LEG_ACTUATORS + RIGHT_LEG_ACTUATORS)

    # Get initial positions
    states = await kos.actuator.get_actuators_state(LEFT_LEG_ACTUATORS + RIGHT_LEG_ACTUATORS)
    start_positions = {state.actuator_id: state.position for state in states.states}

    logger.info("Starting walking sequence...")
    num_steps = 4  # Number of steps to take

    try:
        for step in range(num_steps):
            step_start_time = time.time()
            logger.info("Starting step %d/%d", step + 1, num_steps)

            while time.time() - step_start_time < STEP_DURATION:
                eps_start_time = time.time()
                t = time.time() - step_start_time
                step_phase = (step + t / STEP_DURATION) % 1.0

                # Generate trajectories
                com_x, com_y, com_z = compute_com_trajectory(t, step_phase)
                left_x, left_y, left_z = generate_foot_trajectory(t, step_phase, is_swing_foot=(step % 2 == 0))
                right_x, right_y, right_z = generate_foot_trajectory(t, step_phase, is_swing_foot=(step % 2 == 1))

                # Convert to joint angles using inverse kinematics
                # Note: This is a simplified version - you'll need to implement proper IK
                left_leg_angles = simple_leg_ik(left_x, left_y, left_z, com_x, com_y, com_z)
                right_leg_angles = simple_leg_ik(right_x, right_y, right_z, com_x, com_y, com_z)

                # Send commands to actuators
                commands = []
                for i, angle in enumerate(left_leg_angles):
                    commands.append(
                        {
                            "actuator_id": LEFT_LEG_ACTUATORS[i],
                            "position": math.degrees(angle) + start_positions[LEFT_LEG_ACTUATORS[i]],
                        }
                    )
                for i, angle in enumerate(right_leg_angles):
                    commands.append(
                        {
                            "actuator_id": RIGHT_LEG_ACTUATORS[i],
                            "position": math.degrees(angle) + start_positions[RIGHT_LEG_ACTUATORS[i]],
                        }
                    )

                await kos.actuator.command_actuators(commands)
                await asyncio.sleep(0.01 - (time.time() - eps_start_time))

            logger.info("Completed step %d", step + 1)

    except Exception as e:
        logger.error("Error during walking sequence: %s", e)
        raise

    finally:
        # Return to starting position
        logger.info("Returning to starting position...")
        commands = [{"actuator_id": id, "position": pos} for id, pos in start_positions.items()]
        await kos.actuator.command_actuators(commands)


def simple_leg_ik(foot_x: float, foot_y: float, foot_z: float, com_x: float, com_y: float, com_z: float) -> List[float]:
    """Simplified inverse kinematics for leg joints.

    This is a placeholder that should be replaced with proper IK calculations
    based on the Z-Bot's specific kinematics.
    """
    # This is a very simplified approximation - replace with actual IK
    hip_pitch = math.atan2(foot_x - com_x, com_z - foot_z)
    hip_roll = math.atan2(foot_y - com_y, com_z - foot_z)
    knee_pitch = -math.pi / 6  # Approximate bent knee
    ankle_pitch = -hip_pitch / 2
    ankle_roll = -hip_roll

    return [hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll]


if __name__ == "__main__":
    asyncio.run(main())
