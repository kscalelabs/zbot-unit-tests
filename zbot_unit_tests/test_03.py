import asyncio
import sys
import subprocess
import time
import logging
from pykos import KOS

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_03")

ACTUATOR_MAPPING = {
    "left_shoulder_yaw": 11,
    "left_shoulder_pitch": 12,
    "left_elbow": 13,
    "left_gripper": 14,
    "right_shoulder_yaw": 21,
    "right_shoulder_pitch": 22,
    "right_elbow": 23,
    "right_gripper": 24,
    "left_hip_roll": 31,
    "left_hip_yaw": 32,
    "left_hip_pitch": 33,
    "left_knee": 34,
    "left_ankle": 35,
    "right_hip_roll": 41,
    "right_hip_yaw": 42,
    "right_hip_pitch": 43,
    "right_knee": 44,
    "right_ankle": 45,
}


async def run_robot(kos: KOS, is_real: bool) -> None:
    robot_type = "real robot" if is_real else "simulator"
    logger.info(f"Configuring actuators for {robot_type}")

    for actuator_id in ACTUATOR_MAPPING.values():
        if is_real:
            kp, kd = 32, 32
        else:
            gains = {actuator_id: (100, 50) for actuator_id in ACTUATOR_MAPPING.values()}
            kp, kd = gains.get(actuator_id, (100, 50))

        logger.debug(f"Configuring actuator {actuator_id} with kp={kp}, kd={kd}")
        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=kp,
            kd=kd,
            max_torque=100,
            torque_enabled=True,
        )
    return kos


async def main():
    logger.info("Starting test_03")

    logger.info("Starting simulator server...")
    sim_process = subprocess.Popen(["kos-sim", "zbot-v2-fixed"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)

    try:
        logger.info("Running test on simulator...")
        async with KOS(ip="localhost", port=50051) as sim_kos:
            logger.info("Connected to simulator")

            logger.info("Resetting simulator")
            await sim_kos.sim.reset()
            time.sleep(1)

            logger.info("Configuring simulator robot")
            sim_kos = await run_robot(sim_kos, False)

            logger.info("Connecting to real robot at 192.168.42.1...")
            async with KOS(ip="192.168.42.1") as real_kos:
                logger.info("Connected to real robot")

                logger.info("Configuring real robot")
                real_kos = await run_robot(real_kos, True)

                for kos, name in [(sim_kos, "SIM"), (real_kos, "REAL")]:
                    logger.info(f"[{name}] Getting initial actuator states")
                    response = await kos.actuator.get_actuators_state(list(ACTUATOR_MAPPING.values()))
                    for state in response.states:
                        logger.info(
                            f"[{name}] Initial state of actuator {state.actuator_id}: position={state.position:.2f}, velocity={state.velocity:.2f}"
                        )

                logger.info("Zeroing all joints with velocity...")
                zero_commands = [
                    {"actuator_id": actuator_id, "position": 0, "velocity": 1}
                    for actuator_id in ACTUATOR_MAPPING.values()
                ]
                logger.debug(f"Sending zero commands: {zero_commands}")

                await sim_kos.actuator.command_actuators(zero_commands)
                await real_kos.actuator.command_actuators(zero_commands)
                await asyncio.sleep(2)

                # Test each joint
                for joint_name, actuator_id in ACTUATOR_MAPPING.items():
                    logger.info(f"Testing {joint_name} (ID: {actuator_id})")
                    test_angle = -45  # Full 45 degree angles as requested
                    adjusted_angle = -test_angle if actuator_id in [21, 23, 24] else test_angle

                    # Send command with velocity
                    command = [{"actuator_id": actuator_id, "position": adjusted_angle, "velocity": 10}]
                    logger.debug(f"Sending command to {joint_name}: {command}")

                    await sim_kos.actuator.command_actuators(command)
                    await real_kos.actuator.command_actuators(command)

                    # Wait and check position
                    await asyncio.sleep(2)

                    for kos, name in [(sim_kos, "SIM"), (real_kos, "REAL")]:
                        response = await kos.actuator.get_actuators_state([actuator_id])
                        state = response.states[0]
                        logger.info(
                            f"[{name}] Actuator {actuator_id} ({joint_name}): "
                            f"position={state.position:.2f} (target was {adjusted_angle}), "
                            f"velocity={state.velocity:.2f}"
                        )

                    # Return to zero with velocity
                    command = [{"actuator_id": actuator_id, "position": 0, "velocity": 10}]
                    logger.debug(f"Returning {joint_name} to zero: {command}")

                    await sim_kos.actuator.command_actuators(command)
                    await real_kos.actuator.command_actuators(command)
                    await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
    finally:
        logger.info("Terminating simulator")
        sim_process.terminate()
        sim_process.wait()
        logger.info("Test completed")


if __name__ == "__main__":
    asyncio.run(main())
