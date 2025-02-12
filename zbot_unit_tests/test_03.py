"""
IMPORTANT: Before running this test, start the simulator server with:
    kos-sim zbot-v2
"""

import asyncio
import sys
import subprocess
import time
from pykos import KOS

ACTUATOR_MAPPING = {
    # Arms
    "left_shoulder_yaw": 11,
    "left_shoulder_pitch": 12,
    "left_elbow": 13,
    "left_wrist": 14,
    "right_shoulder_yaw": 21,
    "right_shoulder_pitch": 22,
    "right_elbow": 23,
    "right_wrist": 24,
    # Legs (for zeroing)
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
    # Configure actuators
    for actuator_id in ACTUATOR_MAPPING.values():
        if is_real:
            kp, kd = 32, 32  # Real robot uses 32, 32 for all joints
        else:
            # Simulation gains (higher for better tracking)
            gains = {
                # Arm gains
                11: (50, 5),
                12: (50, 5),
                13: (50, 5),
                14: (30, 3),
                21: (50, 5),
                22: (50, 5),
                23: (50, 5),
                24: (30, 3),
                # Leg gains
                31: (150, 15),
                32: (150, 15),
                33: (150, 15),
                34: (150, 15),
                35: (100, 10),
                41: (150, 15),
                42: (150, 15),
                43: (150, 15),
                44: (150, 15),
                45: (100, 10),
            }
            kp, kd = gains.get(actuator_id, (32, 32))

        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=kp,
            kd=kd,
            max_torque=100,
            torque_enabled=True,
        )
    return kos


async def main():
    # Start the simulator server
    print("Starting simulator server...")
    sim_process = subprocess.Popen(["kos-sim", "zbot-v2"])
    time.sleep(2)  # Wait for simulator to start up

    try:
        print("Running on both simulator and real robot simultaneously...")
        async with KOS(ip="localhost", port=50051) as sim_kos, KOS(ip="10.33.11.170", port=50051) as real_kos:

            # Reset simulator
            await sim_kos.sim.reset()

            # Configure both robots
            sim_kos = await run_robot(sim_kos, False)
            real_kos = await run_robot(real_kos, True)

            # First zero all joints
            print("Zeroing all joints...")
            zero_commands = [{"actuator_id": actuator_id, "position": 0} for actuator_id in ACTUATOR_MAPPING.values()]
            await asyncio.gather(
                sim_kos.actuator.command_actuators(zero_commands), real_kos.actuator.command_actuators(zero_commands)
            )
            await asyncio.sleep(2)  # Wait for joints to zero

            # Test each actuator one by one
            for joint_name, actuator_id in ACTUATOR_MAPPING.items():
                print(f"Testing {joint_name} (ID: {actuator_id})")

                # Set test angles based on joint type
                if actuator_id in [31, 32, 41, 42]:  # leg yaw and roll joints
                    test_angle = 10
                elif actuator_id in [33, 34]:  # left hip pitch and knee
                    test_angle = -45
                elif actuator_id in [11, 13]:  # left shoulder yaw and elbow
                    test_angle = -45
                else:
                    test_angle = 45

                # Move single joint to test angle on both robots simultaneously
                command = [{"actuator_id": actuator_id, "position": test_angle}]
                await asyncio.gather(
                    sim_kos.actuator.command_actuators(command), real_kos.actuator.command_actuators(command)
                )
                await asyncio.sleep(2)  # Wait for movement

                # Return joint to zero on both robots simultaneously
                command = [{"actuator_id": actuator_id, "position": 0}]
                await asyncio.gather(
                    sim_kos.actuator.command_actuators(command), real_kos.actuator.command_actuators(command)
                )
                await asyncio.sleep(1)  # Wait for return to zero

    finally:
        # Clean up: terminate the simulator server
        sim_process.terminate()
        sim_process.wait()


if __name__ == "__main__":
    asyncio.run(main())
