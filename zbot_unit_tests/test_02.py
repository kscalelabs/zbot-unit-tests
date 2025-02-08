"""Uses PyBullet inverse kinematics to control the Z-Bot with configurable limb circles.

Change the SELECTED_LIMB variable to control which limb is animated.
Change the USE_ROBOT_POSITIONS variable to control whether the real robot positions or the simulated positions are used.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING, Any, cast

import colorlogging
import pybullet as p  # type: ignore[import-not-found]
import pybullet_data  # type: ignore[import-untyped]
import pykos
from kscale.web.clients.client import WWWClient

if TYPE_CHECKING:
    from pykos.actuator import ActuatorCommand  # type: ignore[import-not-found]
else:
    # Dummy definition so that ActuatorCommand exists at runtime.
    class ActuatorCommand(dict):
        pass


# ---------------------------
# Global Constants & Settings
# ---------------------------
# Actuator configuration gains, IK limits, and simulation parameters.
ACTUATOR_KP: float = 32.0
ACTUATOR_KD: float = 32.0
TORQUE_ENABLED: bool = True

IK_LOWER_LIMIT: float = -0.967
IK_UPPER_LIMIT: float = 0.967
IK_JOINT_RANGE: float = 1.934
IK_REST_POSE: float = 0
IK_JOINT_DAMPING: float = 0.1

SIM_FORCE: int = 1000
SIM_POSITION_GAIN: float = 0.3
SIM_VELOCITY_GAIN: float = 1

LOOP_DELAY: float = 0.1  # seconds

# ---------------------------
# Actuator and Limb Definitions
# ---------------------------
# Actuator IDs
LEFT_ARM_ACTUATORS: list[int] = [11, 12, 13, 14]
RIGHT_ARM_ACTUATORS: list[int] = [21, 22, 23, 24]
LEFT_LEG_ACTUATORS: list[int] = [31, 32, 33, 34, 35]
RIGHT_LEG_ACTUATORS: list[int] = [41, 42, 43, 44, 45]

ALL_ACTUATORS: list[int] = LEFT_ARM_ACTUATORS + RIGHT_ARM_ACTUATORS + LEFT_LEG_ACTUATORS + RIGHT_LEG_ACTUATORS

# Mapping from actuator IDs to joint names and inversion flags.
ACTUATOR_ID_TO_NAME: dict[int, str] = {
    11: "left_shoulder_yaw",
    12: "left_shoulder_pitch",
    13: "left_elbow",
    14: "left_gripper",
    21: "right_shoulder_yaw",
    22: "right_shoulder_pitch",
    23: "right_elbow",
    24: "right_gripper",
    31: "left_hip_yaw",
    32: "left_hip_roll",
    33: "left_hip_pitch",
    34: "left_knee",
    35: "left_ankle",
    41: "right_hip_yaw",
    42: "right_hip_roll",
    43: "right_hip_pitch",
    44: "right_knee",
    45: "right_ankle",
}

ACTUATOR_ID_TO_INVERSION: dict[int, bool] = {
    11: False,
    12: False,
    13: True,
    14: False,
    21: False,
    22: False,
    23: False,
    24: False,
    31: False,
    32: True,
    33: False,
    34: False,
    35: False,
    41: False,
    42: False,
    43: False,
    44: False,
    45: False,
}


class LimbConfig:
    """Configuration for a robot limb's IK control."""

    def __init__(self, ik_joints: list[int], base_pos: list[float], ee_link: str, radius: float) -> None:
        self.ik_joints: list[int] = ik_joints
        self.base_pos: list[float] = base_pos
        self.ee_link: str = ee_link
        self.radius: float = radius


LIMB_CONFIG: dict[str, LimbConfig] = {
    "left_arm": LimbConfig(
        ik_joints=LEFT_ARM_ACTUATORS,
        base_pos=[0.2, 0.0, -0.1],
        ee_link="FINGER_1",
        radius=0.1,
    ),
    "right_arm": LimbConfig(
        ik_joints=RIGHT_ARM_ACTUATORS,
        base_pos=[-0.2, 0.0, -0.1],
        ee_link="FINGER_1_2",
        radius=0.1,
    ),
    "left_leg": LimbConfig(
        ik_joints=LEFT_LEG_ACTUATORS,
        base_pos=[0.1, 0.0, -0.4],
        ee_link="FOOT",
        radius=0.05,
    ),
    "right_leg": LimbConfig(
        ik_joints=RIGHT_LEG_ACTUATORS,
        base_pos=[-0.1, 0.0, -0.4],
        ee_link="FOOT_2",
        radius=0.05,
    ),
}

# Select which limb to animate.
# Options: "left_arm", "right_arm", "left_leg", "right_leg"
SELECTED_LIMB: str = "right_arm"
# Select whether to use the real robot positions or the simulated positions.
USE_ROBOT_POSITIONS: bool = True

# ---------------------------
# Logger configuration
# ---------------------------
logger: logging.Logger = logging.getLogger(__name__)
# Uncomment the next line to enable debug logging.
# logger.setLevel(logging.DEBUG)


# ---------------------------
# Helper Functions (Module Level)
# ---------------------------
def get_target_position(base_pos: list[float], radius: float, t: float, is_leg: bool = False) -> list[float]:
    """Return a target position that traces a circle in space.

    For legs: circle in x-y plane (top-down view)
    For arms: circle in y-z plane (side view)
    """
    if is_leg:
        return [
            base_pos[0] + radius * math.sin(t),
            base_pos[1] + radius * math.cos(t),
            base_pos[2],
        ]
    else:
        return [
            base_pos[0],
            base_pos[1] + radius * math.cos(t),
            base_pos[2] + radius * math.sin(t),
        ]


def log_joint_info(robot_id: int, joint_name_to_index: dict[str, int], link_name_to_index: dict[str, int]) -> None:
    """Log detailed joint and link information from the PyBullet robot model.

    Also populate joint_name_to_index and link_name_to_index.
    """
    logger.debug("PyBullet Full Joint/Link Information:")
    logger.debug("=====================================")
    # Base link information
    base_name = p.getBodyInfo(robot_id)[0].decode("UTF-8")
    link_name_to_index[base_name] = -1
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode("UTF-8")
        link_name = joint_info[12].decode("UTF-8")
        joint_name_to_index[joint_name] = i
        link_name_to_index[link_name] = i
        logger.debug("Joint %d:", i)
        logger.debug("  Joint Name: %s", joint_name)
        logger.debug("  Link Name: %s", link_name)
        logger.debug("  Joint Axis: %s", joint_info[13])
        logger.debug("  Joint Limits: (%f, %f) (radians)", joint_info[8], joint_info[9])
    logger.debug("Preliminary Actuator ID to Joint Index Mapping:")
    for actuator_id in ALL_ACTUATORS:
        test_joint_name = f"joint_{actuator_id}"
        actual_joint_name = ACTUATOR_ID_TO_NAME.get(actuator_id, "unknown")
        pybullet_index = joint_name_to_index.get(test_joint_name, "not found")
        logger.debug(
            "Actuator %d: using '%s' (should be '%s') -> PyBullet Index: %s",
            actuator_id,
            test_joint_name,
            actual_joint_name,
            pybullet_index,
        )


def create_actuator_joint_mapping(joint_name_to_index: dict[str, int]) -> dict[int, int]:
    """Create a mapping from actuator IDs to PyBullet joint indices."""
    actuator_to_pybullet: dict[int, int] = {}
    logger.debug("Creating actuator-to-PyBullet joint mapping:")
    for actuator_id in ALL_ACTUATORS:
        joint_name = ACTUATOR_ID_TO_NAME[actuator_id]
        if joint_name in joint_name_to_index:
            joint_idx = joint_name_to_index[joint_name]
            actuator_to_pybullet[actuator_id] = joint_idx
            logger.debug("  Actuator %d -> Joint '%s' (PyBullet Index: %d)", actuator_id, joint_name, joint_idx)
        else:
            logger.warning("  Could not find joint '%s' for actuator %d", joint_name, actuator_id)
            actuator_to_pybullet[actuator_id] = -1
    logger.debug("Total mapped actuators: %d", len(actuator_to_pybullet))
    return actuator_to_pybullet


def create_actuator_commands(
    ik_joints: list[int],
    joint_poses: tuple[float, ...],
    actuator_to_pybullet: dict[int, int],
    free_joint_mapping: dict[int, int],
    current_positions: dict[int, float],
) -> list[ActuatorCommand]:
    """Create commands for the real robot from the IK solution."""
    commands: list[ActuatorCommand] = []
    for i, actuator_id in enumerate(ik_joints):
        joint_idx = actuator_to_pybullet[actuator_id]
        if joint_idx != -1 and joint_idx < len(joint_poses):
            ik_index = free_joint_mapping[joint_idx]
            angle_rad = joint_poses[ik_index]
            angle_deg = math.degrees(angle_rad)
            if ACTUATOR_ID_TO_INVERSION[actuator_id]:
                angle_deg = -angle_deg
            # Use a cast to produce an ActuatorCommand from a dict.
            commands.append(cast(ActuatorCommand, {"actuator_id": actuator_id, "position": angle_deg}))
            current_pos: float = current_positions.get(actuator_id, 0)
            logger.debug(
                "Command for Actuator %d (%s): Target: %.2f°  Current: %.2f°%s",
                actuator_id,
                ACTUATOR_ID_TO_NAME[actuator_id],
                angle_deg,
                current_pos,
                " [INVERTED]" if ACTUATOR_ID_TO_INVERSION[actuator_id] else "",
            )
    return commands


# ---------------------------
# ZBotIKController Class
# ---------------------------
class ZBotIKController:
    """Controller for running IK on a selected limb of the Z-Bot."""

    def __init__(self, kos: pykos.KOS, selected_limb: str) -> None:
        """Initialize the IK controller."""
        self.kos: pykos.KOS = kos
        self.selected_limb: str = selected_limb
        self.limb_config: LimbConfig = LIMB_CONFIG[selected_limb]
        self.ik_joints: list[int] = self.limb_config.ik_joints
        self.base_pos: list[float] = self.limb_config.base_pos
        self.ee_link_name: str = self.limb_config.ee_link
        self.radius: float = self.limb_config.radius

        # IK and simulation flags
        self.useNullSpace: bool = True
        self.useOrientation: bool = False
        self.useSimulation: bool = True
        self.ikSolver: int = 0  # Damped Least Squares
        self.use_robot_positions: bool = USE_ROBOT_POSITIONS

        # Track per‐second stats.
        self.commands_this_second: int = 0
        self.iterations_this_second: int = 0
        self.last_log_time: float = time.time()

        # These will be set up in setup_simulation.
        self.physics_client: Any = None
        self.robot_id: int = -1
        self.free_joint_indices: list[int] = []
        self.free_joint_mapping: dict[int, int] = {}
        self.joint_name_to_index: dict[str, int] = {}
        self.link_name_to_index: dict[str, int] = {}
        self.actuator_mapping: dict[int, int] = {}
        self.ee_link_index: int | None = None

    async def setup_simulation(self) -> None:
        """Initialize PyBullet simulation and build joint/link mappings.

        Downloads the URDF model using WWWClient.
        """
        self.physics_client, self.robot_id, self.free_joint_indices = await setup_pybullet()
        self.free_joint_mapping = {joint_idx: idx for idx, joint_idx in enumerate(self.free_joint_indices)}
        log_joint_info(self.robot_id, self.joint_name_to_index, self.link_name_to_index)
        self.actuator_mapping = create_actuator_joint_mapping(self.joint_name_to_index)
        self.ee_link_index = self.link_name_to_index.get(self.ee_link_name, None)
        if self.ee_link_index is None:
            logger.error("End-effector link '%s' not found in robot model.", self.ee_link_name)
            raise ValueError("End-effector link not found.")
        self.log_mapping_info()

    def log_mapping_info(self) -> None:
        """Log the final mapping between actuator IDs and PyBullet joint indices."""
        logger.info("Actuator to PyBullet Joint Mapping:")
        for actuator_id in ALL_ACTUATORS:
            var_joint: str = ACTUATOR_ID_TO_NAME.get(actuator_id, "unknown")
            pybullet_index: int = self.actuator_mapping.get(actuator_id, -1)
            inverted: bool = ACTUATOR_ID_TO_INVERSION.get(actuator_id, False)
            logger.debug(
                "  Actuator %d (%s): PyBullet Joint Index = %d, Inverted: %s",
                actuator_id,
                var_joint,
                pybullet_index,
                inverted,
            )

    def compute_ik_solution(self, target_pos: list[float]) -> tuple[float, ...]:
        """Compute the IK solution for the target position."""
        num_joints: int = len(self.ik_joints)
        lower_limits: list[float] = [IK_LOWER_LIMIT] * num_joints
        upper_limits: list[float] = [IK_UPPER_LIMIT] * num_joints
        joint_ranges: list[float] = [IK_JOINT_RANGE] * num_joints
        rest_poses: list[float] = [IK_REST_POSE] * num_joints

        if self.useNullSpace:
            if self.useOrientation:
                joint_poses = p.calculateInverseKinematics(
                    self.robot_id,
                    endEffectorLinkIndex=self.ee_link_index,
                    targetPosition=target_pos,
                    lowerLimits=lower_limits,
                    upperLimits=upper_limits,
                    jointRanges=joint_ranges,
                    restPoses=rest_poses,
                )
            else:
                joint_poses = p.calculateInverseKinematics(
                    self.robot_id,
                    endEffectorLinkIndex=self.ee_link_index,
                    targetPosition=target_pos,
                    lowerLimits=lower_limits,
                    upperLimits=upper_limits,
                    jointRanges=joint_ranges,
                    restPoses=rest_poses,
                )
        else:
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                endEffectorLinkIndex=self.ee_link_index,
                targetPosition=target_pos,
                solver=self.ikSolver,
            )
        return tuple(joint_poses)

    def update_simulation(self, joint_poses: tuple[float, ...]) -> None:
        """Update the PyBullet simulation for each IK‐controlled joint."""
        for i, actuator_id in enumerate(self.ik_joints):
            # actuator_id is an int.
            joint_idx: int = self.actuator_mapping[actuator_id]
            ik_index: int = self.free_joint_mapping[joint_idx]
            target_angle: float = joint_poses[ik_index]
            logger.debug(
                "Updating simulation for Actuator %d (%s): PyBullet Joint %d -> Target Angle = %.4f rad (ik index: %d)",
                actuator_id,
                ACTUATOR_ID_TO_NAME[actuator_id],
                joint_idx,
                target_angle,
                ik_index,
            )
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_angle,
                targetVelocity=0,
                force=SIM_FORCE,
                positionGain=SIM_POSITION_GAIN,
                velocityGain=SIM_VELOCITY_GAIN,
            )
        p.stepSimulation()

    def log_end_effector_status(self, target_pos: list[float]) -> None:
        """Log the target, actual end‐effector positions, and the distance between them."""
        state = p.getLinkState(self.robot_id, self.ee_link_index)
        actual_pos: tuple[float, ...] = state[0]
        distance: float = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual_pos, target_pos)))
        logger.debug("End Effector Status:")
        logger.debug("  Target Position: %s", [format(x, ".3f") for x in target_pos])
        logger.debug("  Actual Position: %s", [format(x, ".3f") for x in actual_pos])
        logger.debug("  Distance to Target: %.3f meters", distance)

    async def update_robot_visualization(self) -> None:
        """(Optional) Update PyBullet visualization to match real robot joint positions."""
        if self.use_robot_positions:
            states = await self.kos.actuator.get_actuators_state(ALL_ACTUATORS)
            current_positions: dict[int, float] = {state.actuator_id: state.position for state in states.states}
            for actuator_id in ALL_ACTUATORS:
                if actuator_id in current_positions:
                    joint_idx: int = self.actuator_mapping[actuator_id]
                    angle_deg: float = current_positions[actuator_id]
                    if ACTUATOR_ID_TO_INVERSION.get(actuator_id, False):
                        angle_deg = -angle_deg
                    angle_rad: float = math.radians(angle_deg)
                    logger.debug(
                        "Visual Update: Setting PyBullet Joint %d for Actuator %d (%s) to %.4f rad (%.2f°)%s",
                        joint_idx,
                        actuator_id,
                        ACTUATOR_ID_TO_NAME[actuator_id],
                        angle_rad,
                        angle_deg,
                        " [INVERTED]" if ACTUATOR_ID_TO_INVERSION[actuator_id] else "",
                    )
                    p.resetJointState(self.robot_id, joint_idx, angle_rad)

    async def create_and_send_commands(
        self, joint_poses: tuple[float, ...], current_positions: dict[int, float]
    ) -> None:
        """Create commands from the IK solution and (optionally) send them to the robot."""
        commands: list[ActuatorCommand] = create_actuator_commands(
            ik_joints=self.ik_joints,
            joint_poses=joint_poses,
            actuator_to_pybullet=self.actuator_mapping,
            free_joint_mapping=self.free_joint_mapping,
            current_positions=current_positions,
        )
        if commands:
            for cmd in commands:
                # Here we assume cmd is an ActuatorCommand (a dict with keys "actuator_id" and "position")
                actuator_id: int = cmd["actuator_id"]
                pos: float = cmd["position"]
                joint_idx: int = self.actuator_mapping[actuator_id]
                logger.debug(
                    "Command: Actuator %d (%s) -> Set PyBullet Joint %d to %.2f°",
                    actuator_id,
                    ACTUATOR_ID_TO_NAME[actuator_id],
                    joint_idx,
                    pos,
                )
            logger.debug("Sending commands: %s", commands)
            await self.kos.actuator.command_actuators(commands)
            self.commands_this_second += len(commands)

    async def run_loop(self) -> None:
        """Run the main IK control loop."""
        start_time_loop: float = time.time()
        self.last_log_time = start_time_loop

        while True:
            t: float = time.time() - start_time_loop
            # For legs, trace a circle in the x-y plane; for arms, trace a circle in the y-z plane.
            const_is_leg: bool = self.selected_limb in ["left_leg", "right_leg"]
            target_pos: list[float] = get_target_position(self.base_pos, self.radius, t, is_leg=const_is_leg)
            p.addUserDebugPoints([target_pos], [[1, 0, 1]], pointSize=6, lifeTime=1)

            # Compute IK solution and log per‐actuator details.
            joint_poses: tuple[float, ...] = self.compute_ik_solution(target_pos)
            logger.debug("IK Solution:")
            for actuator_id in self.ik_joints:
                joint_idx: int = self.actuator_mapping[actuator_id]
                ik_index: int = self.free_joint_mapping[joint_idx]
                angle_rad: float = joint_poses[ik_index]
                angle_deg: float = math.degrees(angle_rad)
                if ACTUATOR_ID_TO_INVERSION.get(actuator_id, False):
                    angle_deg = -angle_deg
                logger.debug(
                    "  Actuator %d (%s), PyBullet Joint %d: %.4f rad (%.2f°)",
                    actuator_id,
                    ACTUATOR_ID_TO_NAME[actuator_id],
                    joint_idx,
                    angle_rad,
                    angle_deg,
                )

            # Update simulation with the IK solution.
            self.update_simulation(joint_poses)
            self.log_end_effector_status(target_pos)

            # Get current positions from the real robot.
            states = await self.kos.actuator.get_actuators_state(ALL_ACTUATORS)
            current_positions: dict[int, float] = {state.actuator_id: state.position for state in states.states}
            logger.debug("Current Robot Positions: %s", current_positions)
            await self.update_robot_visualization()
            await self.create_and_send_commands(joint_poses, current_positions)

            self.iterations_this_second += 1

            # Log stats every second and reset counters.
            current_time: float = time.time()
            if current_time - self.last_log_time >= 1.0:
                logger.info(
                    "Rate: %d IK solutions/sec, %d commands/sec",
                    self.iterations_this_second,
                    self.commands_this_second,
                )
                self.iterations_this_second = 0
                self.commands_this_second = 0
                self.last_log_time = current_time


# ---------------------------
# PyBullet Setup Function
# ---------------------------
async def setup_pybullet() -> tuple[Any, int, list[int]]:
    """Initialize PyBullet physics simulation and return (physics_client, robot_id, free_joint_indices).

    Downloads the URDF model using WWWClient.
    """
    async with WWWClient() as client:
        urdf_dir = await client.download_and_extract_urdf("zbot-v2")
    try:
        urdf_path = next(urdf_dir.glob("*.urdf"))
    except StopIteration:
        raise ValueError("No URDF file found in the downloaded directory: %s" % urdf_dir)

    physics_client = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    robot_id = p.loadURDF(str(urdf_path), [0, 0, 0], useFixedBase=True)
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0], [0, 0, 0, 1])

    free_joint_indices: list[int] = []
    num_joints: int = p.getNumJoints(robot_id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[2] != p.JOINT_FIXED:
            free_joint_indices.append(i)

    # Draw coordinate axes.
    origin: list[float] = [0, 0, 0]
    length: float = 0.2
    p.addUserDebugLine(origin, [length, 0, 0], [1, 0, 0])
    p.addUserDebugLine(origin, [0, length, 0], [0, 1, 0])
    p.addUserDebugLine(origin, [0, 0, length], [0, 0, 1])

    return physics_client, robot_id, free_joint_indices


async def configure_actuators(kos: pykos.KOS) -> None:
    """Configure all actuators with low gains for smooth motion."""
    for actuator_id in ALL_ACTUATORS:
        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=ACTUATOR_KP,
            kd=ACTUATOR_KD,
            torque_enabled=TORQUE_ENABLED,
        )


# ---------------------------
# Main Async Function
# ---------------------------
async def main() -> None:
    colorlogging.configure()
    logger.warning("Starting PyBullet IK demo")
    try:
        async with pykos.KOS("192.168.42.1") as kos:
            await configure_actuators(kos)
            controller = ZBotIKController(kos, SELECTED_LIMB)
            await controller.setup_simulation()
            await controller.run_loop()
    except Exception:
        logger.exception("Make sure that the Z-Bot is connected over USB and the IP address is accessible.")
        raise
    finally:
        p.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
