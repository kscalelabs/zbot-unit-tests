"""Uses Pybullet inverse kinematics to control the Z-Bot."""

import asyncio
import logging
import math
import time

import colorlogging
import pybullet as p
import pybullet_data
import pykos
from kscale.web.clients.client import WWWClient

logger = logging.getLogger(__name__)

# Actuator IDs from test_01.py
LEFT_ARM_ACTUATORS = [11, 12, 13, 14]
RIGHT_ARM_ACTUATORS = [21, 22, 23, 24]
LEFT_LEG_ACTUATORS = [31, 32, 33, 34, 35]
RIGHT_LEG_ACTUATORS = [41, 42, 43, 44, 45]

ALL_ACTUATORS = LEFT_ARM_ACTUATORS + RIGHT_ARM_ACTUATORS + LEFT_LEG_ACTUATORS + RIGHT_LEG_ACTUATORS


async def main() -> None:
    colorlogging.configure()
    logger.warning("Starting test-03")
    try:
        async with pykos.KOS("192.168.42.1") as kos:
            await ik_movement_test(kos)
    except Exception:
        logger.exception("Make sure that the Z-Bot is connected over USB and the IP address is accessible.")
        raise
    finally:
        p.disconnect()


async def setup_pybullet() -> tuple[int, int]:
    """Initialize PyBullet physics simulation."""
    # Downloads the URDF model.
    async with WWWClient() as client:
        urdf_dir = await client.download_and_extract_urdf("zbot-v2")

    try:
        urdf_path = next(urdf_dir.glob("*.urdf"))
    except StopIteration:
        raise ValueError(f"No URDF file found in the downloaded directory: {urdf_dir}")

    # Connect to PyBullet
    physics_client = p.connect(p.DIRECT)  # Use DIRECT mode since we only need IK calculations
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load robot model - using KUKA as placeholder, should be replaced with Z-Bot URDF
    p.setGravity(0, 0, 0)
    robot_id = p.loadURDF(str(urdf_path), [0, 0, 0])
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0], [0, 0, 0, 1])

    return physics_client, robot_id


async def configure_actuators(kos: pykos.KOS, actuator_ids: list[int]) -> None:
    """Configure the specified actuators."""
    for actuator_id in actuator_ids:
        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=32.0,
            kd=32.0,
            torque_enabled=True,
        )


async def ik_movement_test(kos: pykos.KOS) -> None:
    """Run inverse kinematics-based movement test."""
    # Setup PyBullet
    _, robot_id = await setup_pybullet()

    # Configure all actuators
    await configure_actuators(kos, ALL_ACTUATORS)

    # Get initial states
    states = await kos.actuator.get_actuators_state(ALL_ACTUATORS)
    start_positions = {state.actuator_id: state.position for state in states.states}
    missing_ids = set(ALL_ACTUATORS) - set(start_positions.keys())
    if missing_ids:
        raise ValueError(f"Actuator IDs {missing_ids} not found in start positions")

    # Movement parameters
    duration = 10.0  # seconds
    start_time = time.time()

    while time.time() - start_time < duration:
        t = time.time() - start_time

        # Generate target end effector position (circular motion)
        pos = [-0.4, 0.2 * math.cos(t), 0.2 * math.sin(t)]
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])

        # Calculate inverse kinematics
        joint_poses = p.calculateInverseKinematics(
            robot_id,
            endEffectorLinkIndex=6,  # This should match Z-Bot's end effector link
            targetPosition=pos,
            targetOrientation=orn,
            maxNumIterations=100,
            residualThreshold=0.01,
        )

        # Map IK solution to Z-Bot actuators and send commands
        # Note: This mapping needs to be adjusted based on Z-Bot's kinematics
        commands = []
        for i, actuator_id in enumerate(LEFT_ARM_ACTUATORS):  # Starting with left arm as example
            if i < len(joint_poses):
                commands.append(
                    {"actuator_id": actuator_id, "position": math.degrees(joint_poses[i])}  # Convert radians to degrees
                )

        if commands:
            await kos.actuator.command_actuators(commands)

        await asyncio.sleep(0.01)  # Control rate

    logger.info("IK movement test completed")


if __name__ == "__main__":
    asyncio.run(main())
