"""Demonstrates PyBullet inverse kinematics with Z-Bot."""

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

# Just using left arm for demonstration
LEFT_ARM_ACTUATORS = [11, 12, 13, 14]
HAND_END_EFFECTOR_LINK_NAME = "FINGER_1"


async def main() -> None:
    colorlogging.configure()
    logger.warning("Starting PyBullet IK demo")
    try:
        async with pykos.KOS("10.33.11.170") as kos:
            await ik_demo(kos)
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

    # Connect to PyBullet with GUI
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Configure camera for better view
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

    # Load robot model
    p.setGravity(0, 0, 0)
    robot_id = p.loadURDF(str(urdf_path), [0, 0, 0])
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0], [0, 0, 0, 1])

    return physics_client, robot_id


async def configure_actuators(kos: pykos.KOS) -> None:
    """Configure the arm actuators with low gains for smooth motion."""
    for actuator_id in LEFT_ARM_ACTUATORS:
        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=4.0,  # Very low gains for gentle motion
            kd=4.0,
            torque_enabled=True,
        )


async def ik_demo(kos: pykos.KOS) -> None:
    """Demonstrate inverse kinematics with PyBullet and real robot."""
    # Setup PyBullet
    _, robot_id = await setup_pybullet()

    # Get joint mapping
    link_name_to_index = {p.getBodyInfo(robot_id)[0].decode("UTF-8"): -1}
    joint_name_to_index = {}
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode("UTF-8")
        link_name = joint_info[12].decode("UTF-8")
        joint_name_to_index[joint_name] = i
        link_name_to_index[link_name] = i
        logger.info(f"Joint {i}: {joint_name}")

    hand_link_index = link_name_to_index[HAND_END_EFFECTOR_LINK_NAME]

    # Configure actuators
    await configure_actuators(kos)

    # Store joint mapping for left arm
    joint_indices = []
    logger.info("Joint mapping for left arm:")
    for i, actuator_id in enumerate(LEFT_ARM_ACTUATORS):
        joint_name = f"joint_{actuator_id}"
        if joint_name in joint_name_to_index:
            joint_idx = joint_name_to_index[joint_name]
            joint_indices.append(joint_idx)
            logger.info(f"Actuator {actuator_id} -> Joint {joint_idx} ({joint_name})")
        else:
            logger.warning(f"Could not find joint for actuator {actuator_id}")
            joint_indices.append(i)

    start_time = time.time()
    try:
        while True:  # Run until Ctrl+C
            t = time.time() - start_time

            # Simple up-down motion
            height = 0.005 * math.sin(t / 4)  # Very small, slow motion
            pos = [-0.2, 0, height]  # Just moving up and down
            orn = p.getQuaternionFromEuler([0, -math.pi, 0])

            # Calculate IK
            joint_poses = p.calculateInverseKinematics(
                robot_id,
                endEffectorLinkIndex=hand_link_index,
                targetPosition=pos,
                targetOrientation=orn,
                maxNumIterations=200,
                residualThreshold=0.01,
            )

            # Get current robot state
            states = await kos.actuator.get_actuators_state(LEFT_ARM_ACTUATORS)
            current_positions = {state.actuator_id: state.position for state in states.states}

            # Send commands and update simulation
            commands = []
            for i, actuator_id in enumerate(LEFT_ARM_ACTUATORS):
                if i < len(joint_poses):
                    joint_idx = joint_indices[i]
                    angle_radians = joint_poses[joint_idx]
                    angle_degrees = math.degrees(angle_radians)

                    # Limit range for safety
                    angle_degrees = max(-20, min(20, angle_degrees))  # Even smaller range
                    angle_radians = math.radians(angle_degrees)

                    commands.append({"actuator_id": actuator_id, "position": angle_degrees})

                    # Update PyBullet (in radians)
                    p.setJointMotorControl2(
                        robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=angle_radians,
                        maxVelocity=0.3,  # Very slow movement
                    )

                    # Log positions
                    current_pos = current_positions.get(actuator_id, 0)
                    logger.info(
                        f"Actuator {actuator_id}: Target: {angle_degrees:.2f}° Current: {current_pos:.2f}° "
                        f"(PyBullet: {angle_radians:.3f} rad)"
                    )

            if commands:
                await kos.actuator.command_actuators(commands)

            # Step simulation
            for _ in range(5):
                p.stepSimulation()
            await asyncio.sleep(0.1)  # Slower rate for gentler motion

    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    except Exception as e:
        logger.exception("Error during demo")
        raise

    logger.info("IK demo completed")


if __name__ == "__main__":
    asyncio.run(main())
